#!/usr/bin/env python3
"""Markdown -> .docx (python-docx, real tables) -> .pdf (LibreOffice).
Usage: .venv/bin/python tools/md_to_pdf.py results/corpus_overview.md
Handles: # / ## headings, paragraphs, - bullets, | tables |, ``` code blocks,
**bold**, *italic*, `code`. Landscape for wide tables is not attempted."""
import re, subprocess, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

src = Path(sys.argv[1]); md = src.read_text().splitlines()
doc = Document()
for s in doc.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(1.8); s.top_margin = s.bottom_margin = Cm(1.8)
st = doc.styles["Normal"]; st.font.name = "Liberation Serif"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Serif")
for name, size in [("Heading 1", 18), ("Heading 2", 13)]:
    h = doc.styles[name]; h.font.name = "Liberation Serif"; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Serif")
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
def add_runs(par, text, size=None):
    for part in INLINE.split(text):
        if not part: continue
        if part.startswith("**"): r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"): r = par.add_run(part[1:-1]); r.font.name = "Liberation Mono"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Mono"); r.font.size = Pt((size or 10.5) - 1.5)
        elif part.startswith("*"): r = par.add_run(part[1:-1]); r.italic = True
        else: r = par.add_run(part)
        if size and not (part.startswith("`")): r.font.size = Pt(size)
def shade(cell, hexcolor):
    tcPr = cell._element.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)
def add_table(rows):
    split = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
    H = split(rows[0]); B = [split(r) for r in rows[1:] if not set(r.replace("|", "").strip()) <= set("-: ")]
    ncol = len(H); small = 8.5 if ncol >= 6 else 9
    t = doc.add_table(rows=1 + len(B), cols=ncol); t.style = "Table Grid"
    # column widths proportional to content length (sqrt-damped), summing to the text width
    lens = [max(len(H[j]), *(len(r[j]) if j < len(r) else 0 for r in B)) for j in range(ncol)]
    w = [max(10.0, min(l, 60)) ** 0.5 for l in lens]; total = sum(w); text_w = 17.4
    widths = [Cm(text_w * x / total) for x in w]
    t.autofit = False
    tblPr = t._tbl.tblPr; lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    for j, col in enumerate(t.columns): col.width = widths[j]
    for row in t.rows:
        for j, c in enumerate(row.cells): c.width = widths[j]
    for j, h in enumerate(H):
        c = t.rows[0].cells[j]; c.text = ""; p = c.paragraphs[0]; add_runs(p, h, small)
        for r in p.runs: r.bold = True
        shade(c, "EEEEEE")
    for i, row in enumerate(B, 1):
        for j in range(ncol):
            c = t.rows[i].cells[j]; c.text = ""; add_runs(c.paragraphs[0], row[j] if j < len(row) else "", small)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
i = 0; tbl = []; code = False; cbuf = []
while i < len(md):
    line = md[i]
    if line.startswith("```"):
        if code:
            p = doc.add_paragraph(); r = p.add_run("\n".join(cbuf)); r.font.name = "Liberation Mono"; r.font.size = Pt(8.5)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Mono"); cbuf = []
        code = not code; i += 1; continue
    if code: cbuf.append(line); i += 1; continue
    if line.startswith("|"):
        tbl.append(line); i += 1
        if i == len(md) or not md[i].startswith("|"): add_table(tbl); tbl = []
        continue
    if line.startswith("# "): doc.add_heading(line[2:], level=1)
    elif line.startswith("## "): doc.add_heading(line[3:], level=2)
    elif line.startswith("- "):
        p = doc.add_paragraph(style="List Bullet"); add_runs(p, line[2:]); p.paragraph_format.space_after = Pt(2)
    elif line.strip():
        p = doc.add_paragraph(); add_runs(p, line); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(5)
    i += 1
docx_path = src.with_suffix(".docx"); doc.save(docx_path)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", str(docx_path), "--outdir", str(src.parent)], check=True, capture_output=True, timeout=600)
print("wrote", docx_path, "and", src.with_suffix(".pdf"))
