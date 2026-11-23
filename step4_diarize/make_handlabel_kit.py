#!/usr/bin/env python3
"""Build a browser labelling kit for the hand-label episodes.

For every data/output/diarization/handlabel_<episode>.csv this writes, under
data/output/diarization/handlabel_kit/:

  index.html            list of episodes with progress
  <episode>.html        one page per episode: the episode audio, one row per
                        Whisper segment with a play button that seeks to that
                        segment, H / G / A / X buttons, keyboard shortcuts,
                        autosave in the browser, and "Download CSV" which
                        writes the filled handlabel_<episode>.csv
  audio/<episode>.mp3   the archived audio re-encoded as 32 kb/s mono mp3
                        (plays in every browser; ~12 MB per hour)

Serve the folder and open it through VS Code's forwarded port:

    cd data/output/diarization/handlabel_kit && python3 -m http.server 8765

Then drop each downloaded CSV over the matching handlabel_<episode>.csv in
data/output/diarization/ and run `pilot_eval.py score`.

Labels: H = host, G = guest or any other voice, A = ad / clip / music,
X = unsure. Rows can stay blank; the scorer skips them. Every ~5th row is
marked "suggested" so a 25-row sample per episode is enough if time is short.
"""
from __future__ import annotations

import csv
import html
import json
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data/output/diarization"
KIT = OUT_DIR / "handlabel_kit"
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
CORPUS_SHOWS = PROJECT_ROOT / "data/output/corpus_shows.csv"
AUDIO_SUFFIXES = (".opus", ".mp3", ".m4a", ".wav")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 :root{--bg:#fbfbf8;--fg:#14213d;--muted:#6b7280;--line:#e5e7eb;--h:#2a78d6;--g:#e34948;--a:#eb6834;--x:#9ca3af;--sel:#eef4ff}
 body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
 header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
 header h1{font-size:17px;margin:0 0 6px}
 header .meta{color:var(--muted);font-size:13px}
 audio{width:100%;margin-top:6px}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:6px;font-size:13px}
 button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:6px;padding:4px 10px;cursor:pointer}
 button:hover{background:#f3f4f6}
 table{border-collapse:collapse;width:100%}
 td{border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}
 tr.cur{background:var(--sel)}
 td.t{white-space:nowrap;color:var(--muted);font-variant-numeric:tabular-nums;width:110px}
 td.play{width:40px}
 td.lab{white-space:nowrap;width:170px}
 .lab button{width:34px;padding:4px 0;margin-right:3px;font-weight:600}
 .lab button.on.H{background:var(--h);color:#fff;border-color:var(--h)}
 .lab button.on.G{background:var(--g);color:#fff;border-color:var(--g)}
 .lab button.on.A{background:var(--a);color:#fff;border-color:var(--a)}
 .lab button.on.X{background:var(--x);color:#fff;border-color:var(--x)}
 tr.sug td.t::after{content:" •";color:var(--h)}
 .help{color:var(--muted);font-size:13px;padding:8px 16px}
 .done{color:var(--h);font-weight:600}
 @media(prefers-color-scheme:dark){:root{--bg:#14161c;--fg:#f1f1ee;--muted:#9aa0ab;--line:#2a2e38;--sel:#1f2a44}button{background:#1b1e26;color:var(--fg)}button:hover{background:#262a35}}
</style></head><body>
<header>
 <h1>__TITLE__</h1>
 <div class="meta">__META__</div>
 <audio id="au" controls preload="metadata" src="audio/__EID__.mp3"></audio>
 <div class="bar">
  <span>Progress: <b id="prog">0</b> / __N__ labelled</span>
  <button onclick="saveServer()" style="font-weight:600">Save to server</button> <span id="savemsg" style="color:var(--muted)"></span>
  <button onclick="dl()">Download CSV</button>
  <button onclick="if(confirm('Clear all labels for this episode?')){labels={};save();render()}">Clear</button>
  <label><input type="checkbox" id="sugonly" onchange="render()"> suggested rows only</label>
  <label><input type="checkbox" id="stopend" checked> stop at segment end</label>
  <span>speed: <button onclick="setRate(1)" id="r1">1×</button><button onclick="setRate(1.5)" id="r1.5">1.5×</button><button onclick="setRate(2)" id="r2">2×</button></span>
  <a href="index.html">← all episodes</a>
 </div>
</header>
<div class="help">Click ▶ on a row (or press <b>space</b> for the current row), then <b>h</b> host · <b>g</b> guest / other voice · <b>a</b> ad, clip or music · <b>x</b> unsure. <b>j</b>/<b>k</b> move down/up · <b>1</b>/<b>2</b> playback speed 1× / 2×. Rows marked • are the suggested 25-row sample. Labels autosave in this browser; download the CSV when done.</div>
<table id="tbl"></table>
<script>
const SEGS=__SEGS__; const EID="__EID__", SID="__SID__";
const KEY="handlabel:"+EID+":"+SEGS.length, MIGRATE=__MIGRATE__; let labels={};
try{labels=JSON.parse(localStorage.getItem(KEY)||"{}");
 if(!Object.keys(labels).length&&MIGRATE){const old=JSON.parse(localStorage.getItem("handlabel:"+EID)||"{}");if(Object.keys(old).length){labels=old;localStorage.setItem(KEY,JSON.stringify(labels))}}}catch(e){}
let cur=0, stopAt=null; const au=document.getElementById("au");
let rate=1; try{rate=parseFloat(localStorage.getItem("handlabel:rate")||"1")||1}catch(e){}
function setRate(r){rate=r;au.playbackRate=r;try{localStorage.setItem("handlabel:rate",String(r))}catch(e){}
 ["r1","r1.5","r2"].forEach(id=>{const b=document.getElementById(id);if(b)b.style.fontWeight=(id==="r"+r)?"700":"400"})}
au.addEventListener("loadedmetadata",()=>{au.playbackRate=rate}); au.addEventListener("play",()=>{au.playbackRate=rate});
setTimeout(()=>setRate(rate),0);
// labels saved on the server win when the browser has none (new browser, rebuilt kit)
fetch("/labels/"+EID,{cache:"no-store"}).then(r=>r.json()).then(srv=>{const n=Object.keys(srv).length;
 if(n&&Object.keys(labels).length<n){labels=srv;save();render();document.getElementById("savemsg").textContent="loaded "+n+" labels from server"}}).catch(e=>{});
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;if(e.key==="1")setRate(1);if(e.key==="2")setRate(2);if(e.key==="3")setRate(1.5)});
function save(){try{localStorage.setItem(KEY,JSON.stringify(labels))}catch(e){}}
function fmt(s){const m=Math.floor(s/60),x=Math.floor(s%60);return m+":"+(x<10?"0":"")+x}
function play(i){cur=i;const s=SEGS[i];au.currentTime=s.start;stopAt=document.getElementById("stopend").checked?s.end:null;au.play();render()}
au.addEventListener("timeupdate",()=>{if(stopAt!==null&&au.currentTime>=stopAt){au.pause();stopAt=null}});
function setLab(i,l){if(labels[i]===l)delete labels[i];else labels[i]=l;save();render()}
function render(){
 const only=document.getElementById("sugonly").checked; const t=document.getElementById("tbl"); let h="";
 SEGS.forEach((s,i)=>{ if(only&&!s.sug)return; const l=labels[i]||"";
  h+=`<tr class="${i===cur?'cur':''} ${s.sug?'sug':''}" id="r${i}"><td class="play"><button onclick="play(${i})">▶</button></td><td class="t">${fmt(s.start)}–${fmt(s.end)}</td><td>${s.text}</td><td class="lab">`+
   ["H","G","A","X"].map(k=>`<button class="${k} ${l===k?'on':''}" onclick="setLab(${i},'${k}')">${k}</button>`).join("")+`</td></tr>`});
 t.innerHTML=h; const n=Object.keys(labels).length; const p=document.getElementById("prog"); p.textContent=n; p.className=n>=25?"done":"";
}
document.addEventListener("keydown",e=>{ if(e.target.tagName==="INPUT")return; const k=e.key.toLowerCase();
 if(k===" "){e.preventDefault();play(cur)} else if("hgax".includes(k)&&k.length===1){setLab(cur,k.toUpperCase());next(1)}
 else if(k==="j"){next(1)} else if(k==="k"){next(-1)} });
function next(d){const only=document.getElementById("sugonly").checked; let i=cur; do{i+=d}while(i>=0&&i<SEGS.length&&only&&!SEGS[i].sug); if(i>=0&&i<SEGS.length){cur=i;render();const r=document.getElementById("r"+i);if(r)r.scrollIntoView({block:"center"})}}
function csv(){ const esc=v=>'"'+String(v).replace(/"/g,'""')+'"';
 let out="show_id,episode_id,start,end,text,label\\n";
 SEGS.forEach((s,i)=>{out+=[SID,EID,s.start,s.end,esc(s.raw),labels[i]||""].join(",")+"\\n"}); return out; }
function dl(){ const b=new Blob([csv()],{type:"text/csv"}); const a=document.createElement("a"); a.href=URL.createObjectURL(b); a.download="handlabel_"+EID+".csv"; a.click(); }
function saveServer(){ const m=document.getElementById("savemsg"); m.textContent="saving…";
 fetch("/save/"+EID,{method:"POST",headers:{"Content-Type":"text/csv"},body:csv()}).then(r=>r.text().then(t=>{m.textContent=r.ok?("saved "+Object.keys(labels).length+" labels ✓"):("error: "+t)})).catch(e=>{m.textContent="error: "+e}); }
render();
</script></body></html>
"""

INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hand-label episodes</title>
<style>body{margin:0;padding:20px;font:15px/1.5 system-ui,sans-serif;color:#14213d;background:#fbfbf8}
table{border-collapse:collapse}td,th{padding:8px 12px;border-bottom:1px solid #e5e7eb;text-align:left}a{color:#2a78d6}
@media(prefers-color-scheme:dark){body{background:#14161c;color:#f1f1ee}td,th{border-color:#2a2e38}}</style></head><body>
<h1 style="font-size:20px">Hand-label episodes (__N__)</h1>
<p>Open an episode, label the rows marked • (about 25 per episode, 5–10 minutes with audio), download the CSV, and put it in <code>data/output/diarization/</code> replacing the file of the same name. H host · G guest/other · A ad/clip · X unsure.</p>
<table><tr><th>show</th><th>episode</th><th>length</th><th>segments</th><th>labelled (saved on server · in this browser)</th></tr>__ROWS__</table>
<script>
document.querySelectorAll("[data-eid]").forEach(td=>{try{const n=Object.keys(JSON.parse(localStorage.getItem("handlabel:"+td.dataset.eid+":"+td.dataset.n)||"{}")).length;td.textContent=n+" in browser"}catch(e){}});
fetch("/status",{cache:"no-store"}).then(r=>r.json()).then(st=>{document.querySelectorAll("[data-eid]").forEach(td=>{const n=st[td.dataset.eid]||0;td.innerHTML=(n?"<b>"+n+" saved</b>":"0 saved")+" · "+td.textContent})}).catch(e=>{});
</script>
</body></html>
"""


def find_audio(show_id: str, episode_id: str) -> Path | None:
    for suf in AUDIO_SUFFIXES:
        p = ARCHIVE_DIR / show_id / f"{episode_id}{suf}"
        if p.exists():
            return p
    return None


def encode_mp3(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    tmp = dst.with_suffix(".mp3.part")
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3", str(tmp)], check=True, timeout=1800)
    tmp.rename(dst)


def main() -> None:
    (KIT / "audio").mkdir(parents=True, exist_ok=True)
    shows = pd.read_csv(CORPUS_SHOWS, dtype=str).set_index("show_id")
    rows_html = []
    sheets = sorted(OUT_DIR.glob("handlabel_*.csv"))
    for sheet in sheets:
        lab = pd.read_csv(sheet, dtype={"show_id": str, "episode_id": str})
        if lab.empty:
            continue
        sid, eid = lab.show_id.iloc[0], lab.episode_id.iloc[0]
        audio = find_audio(sid, eid)
        if audio is None:
            print("no audio for", eid)
            continue
        encode_mp3(audio, KIT / "audio" / f"{eid}.mp3")
        n = len(lab)
        step = max(1, round(n / 25))
        segs = [{"start": round(float(r.start), 2), "end": round(float(r.end), 2),
                 "text": html.escape(str(r.text)), "raw": str(r.text), "sug": (i % step == 0)}
                for i, r in enumerate(lab.itertuples())]
        show_name = shows.show_name.get(sid, sid)
        dur = float(lab.end.max())
        title = f"{show_name}"
        meta = f"episode {eid} · {dur / 60:.0f} min · {n} segments · show {sid}"
        page = (PAGE.replace("__TITLE__", html.escape(title)).replace("__META__", html.escape(meta))
                .replace("__EID__", eid).replace("__SID__", sid).replace("__N__", str(n))
                .replace("__SEGS__", json.dumps(segs, ensure_ascii=False))
                .replace("__MIGRATE__", "false" if (PROJECT_ROOT / "data/transcripts_v2" / sid / f"{eid}.json").exists() else "true"))
        (KIT / f"{eid}.html").write_text(page)
        rows_html.append(f'<tr><td>{html.escape(show_name)}</td><td><a href="{eid}.html">{eid}</a></td>'
                         f'<td>{dur / 60:.0f} min</td><td>{n}</td><td data-eid="{eid}" data-n="{n}">–</td></tr>')
        print("built", eid, show_name)
    (KIT / "index.html").write_text(INDEX.replace("__N__", str(len(rows_html))).replace("__ROWS__", "\n".join(rows_html)))
    (KIT / "serve.sh").write_text("#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\" && python3 -m http.server ${1:-8765}\n")
    print(f"kit: {KIT}/index.html ({len(rows_html)} episodes). Serve with: bash {KIT}/serve.sh")


if __name__ == "__main__":
    main()
