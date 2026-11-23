#!/usr/bin/env python3
"""Time each stage of the diarization pipeline on one archived episode.

Prints decode, segmentation, embedding and clustering wall time for a given
configuration, so speed work targets the stage that actually costs.

    HF_TOKEN=... ../whisperx-env/bin/python step4_diarize/profile_stages.py SHOW_ID EPISODE_ID [--stock] [--stride 0.3] [--model ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accelerated_pipeline import AccelConfig, build_pipeline, describe  # noqa: E402
from diarize import find_audio, load_waveform  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("show_id"); ap.add_argument("episode_id")
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--stride", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--embed-mode", default="active")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--mcs-fraction", type=float, default=0.01)
    ap.add_argument("--model", default=AccelConfig.model)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = AccelConfig.stock(args.model, args.device) if args.stock else AccelConfig(
        model=args.model, stride=args.stride, batch_size=args.batch, embed_mode=args.embed_mode,
        mcs_fraction=args.mcs_fraction or None, half=not args.fp32, device=args.device)
    pipeline = build_pipeline(cfg, token=os.environ.get("HF_TOKEN"))
    print("config:", describe(pipeline, cfg))

    t0 = time.monotonic()
    waveform, sr = load_waveform(find_audio(args.show_id, args.episode_id))
    t_decode = time.monotonic() - t0
    duration = waveform.shape[1] / sr

    marks: dict[str, float] = {}
    last = {"t": time.monotonic(), "name": None}

    def hook(name, artefact=None, file=None, total=None, completed=None, **kw):
        # pyannote calls hook(step_name, ...) once per step, plus progress ticks with completed/total
        now = time.monotonic()
        if name != last["name"]:
            if last["name"] is not None:
                marks[last["name"]] = marks.get(last["name"], 0.0) + (now - last["t"])
            last["name"], last["t"] = name, now
        elif completed is not None and completed == total:
            marks[name] = marks.get(name, 0.0) + (now - last["t"])
            last["t"] = now

    t1 = time.monotonic()
    out = pipeline({"waveform": waveform, "sample_rate": sr, "uri": args.episode_id}, hook=hook)
    t_total = time.monotonic() - t1
    marks[last["name"]] = marks.get(last["name"], 0.0) + (time.monotonic() - last["t"])

    ann = out.speaker_diarization if hasattr(out, "speaker_diarization") else out
    print(f"audio {duration:.0f}s | decode {t_decode:.1f}s | pipeline {t_total:.1f}s | "
          f"rtf {duration / (t_decode + t_total):.0f}x (pipeline only {duration / t_total:.0f}x) | speakers {len(ann.labels())}")
    for name, secs in sorted(marks.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {secs:7.1f}s  {100 * secs / t_total:5.1f}%")


if __name__ == "__main__":
    main()
