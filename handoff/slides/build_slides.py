"""Build the presentation deck for 'Talking at the Audience' as an editable .pptx.
Every number is taken from the paper précis as of 10 Sept 2026. Speaker notes on every slide."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------- palette (matches the précis) ----------
PAPER = RGBColor(0xF1, 0xF3, 0xF5); INK = RGBColor(0x14, 0x17, 0x1C); INK2 = RGBColor(0x43, 0x4A, 0x55)
INK3 = RGBColor(0x73, 0x7B, 0x87); RULE = RGBColor(0xD6, 0xDB, 0xE1); ACCENT = RGBColor(0x7A, 0x2E, 0x2E)
DATA = RGBColor(0x3E, 0x52, 0x66); DATA_SOFT = RGBColor(0xE4, 0xEA, 0xF0); RAISED = RGBColor(0xFA, 0xFB, 0xFC)
DARK = RGBColor(0x14, 0x17, 0x1C); DARK_INK = RGBColor(0xE7, 0xE9, 0xEC); DARK_ACC = RGBColor(0xCF, 0x72, 0x72)
SERIF = "Georgia"; SANS = "Calibri"

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
M = Inches(0.8)   # margin

def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color

def box(slide, x, y, w, h, fill=None, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.shadow.inherit = False
    if fill is None: s.fill.background()
    else: s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(0.75)
    return s

def text(slide, x, y, w, h, runs, size=18, color=INK, font=SANS, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, line_spacing=1.15, space_after=6):
    """runs: str, or list of paragraphs; each paragraph str or list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    tf.vertical_anchor = anchor; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.line_spacing = line_spacing; para.space_after = Pt(space_after)
        segs = p if isinstance(p, list) else [(p, {})]
        for seg, o in segs:
            r = para.add_run(); r.text = seg; f = r.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False); f.color.rgb = o.get("color", color)
    return tb

def bullets(slide, x, y, w, h, items, size=18, color=INK, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(gap); para.line_spacing = 1.12
        sub = isinstance(it, tuple) and it[0] == ">"
        txt = it[1] if sub else it
        lead = para.add_run(); lead.text = ("      – " if sub else "•  "); lead.font.color.rgb = ACCENT if not sub else INK3
        lead.font.size = Pt(size - (2 if sub else 0)); lead.font.name = SANS
        segs = txt if isinstance(txt, list) else [(txt, {})]
        for seg, o in segs:
            r = para.add_run(); r.text = seg; r.font.name = SANS; r.font.size = Pt(o.get("size", size - (2 if sub else 0)))
            r.font.bold = o.get("bold", False); r.font.italic = o.get("italic", False); r.font.color.rgb = o.get("color", color if not sub else INK2)
    return tb

def table(slide, x, y, w, rows, col_w=None, size=13, header=True, hi_cols=(), row_h=Inches(0.36), dark=False):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, x, y, w, row_h * nr); t = shp.table
    if col_w:
        for i, cw in enumerate(col_w): t.columns[i].width = cw
    for r in range(nr):
        t.rows[r].height = row_h
        for c in range(nc):
            cell = t.cell(r, c); cell.margin_left = cell.margin_right = Inches(0.08); cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.fill.solid(); cell.fill.fore_color.rgb = (RGBColor(0x1F,0x29,0x32) if (header and r == 0) else (RGBColor(0x1B,0x1F,0x25) if r % 2 else RGBColor(0x17,0x1B,0x21))) if dark else (DATA_SOFT if (header and r == 0) else (RAISED if r % 2 else PAPER))
            tf = cell.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.text = ""; run = p.add_run(); run.text = str(rows[r][c])
            f = run.font; f.name = SANS; f.size = Pt(size)
            if header and r == 0: f.bold = True; f.color.rgb = (RGBColor(0xB2,0xB8,0xC1) if dark else INK2); f.size = Pt(size - 1)
            else:
                f.color.rgb = (DARK_ACC if dark else ACCENT) if c in hi_cols else (DARK_INK if dark else INK); f.bold = c in hi_cols
            if c > 0: p.alignment = PP_ALIGN.RIGHT if (header and r > 0 and rows[r][c][:1] in "+−-0123456789.[<≈r") else PP_ALIGN.LEFT
    return shp

def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s

def kicker(slide, s, dark=False):
    text(slide, M, Inches(0.45), Inches(8), Inches(0.3), s.upper(), size=11, color=(DARK_ACC if dark else ACCENT), bold=True)

def title(slide, s, dark=False, y=Inches(0.8), size=34):
    two = len(s) > 52
    text(slide, M, y, W - 2 * M, Inches(1.2), s, size=(size if not two else min(size, 30)), font=SERIF, color=(DARK_INK if dark else INK), line_spacing=1.05)
    box(slide, M, y + (Inches(1.25) if two else Inches(0.75)), Inches(0.9), Emu(28000), fill=(DARK_ACC if dark else ACCENT))

def footer(slide, n, dark=False):
    c = INK3
    text(slide, M, H - Inches(0.5), Inches(9), Inches(0.3), "Talking at the Audience  ·  Yang & Alvarez  ·  Linde Center, Caltech", size=10, color=c)
    text(slide, W - M - Inches(1), H - Inches(0.5), Inches(1), Inches(0.3), str(n), size=10, color=c, align=PP_ALIGN.RIGHT)

def stat_row(slide, y, stats, cols=4):
    gw = (W - 2 * M - Inches(0.3) * (cols - 1)) / cols
    for i, (v, k) in enumerate(stats):
        x = M + i * (gw + Inches(0.3))
        box(slide, x, y, gw, Inches(1.35), fill=RAISED, line=RULE)
        text(slide, x + Inches(0.2), y + Inches(0.15), gw - Inches(0.4), Inches(0.6), v, size=30, font=SERIF, color=DATA)
        text(slide, x + Inches(0.2), y + Inches(0.78), gw - Inches(0.4), Inches(0.55), k, size=11, color=INK2, line_spacing=1.05)

def claim(slide, y, s, w=None):
    w = w or (W - 2 * M)
    box(slide, M, y, Emu(36000), Inches(0.9), fill=ACCENT)
    text(slide, M + Inches(0.25), y, w - Inches(0.25), Inches(0.9), s, size=20, font=SERIF, color=ACCENT, anchor=MSO_ANCHOR.MIDDLE)


from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
def bar_chart(slide, x, y, w, h, cats, series, horizontal=True, colors=None, fmt='+0.00;-0.00;0.00', dark=False,
              legend=None, size=11, gap=60, overlap=-10, axis_max=None, axis_min=None):
    """series: list of (name, values). Editable native chart; labels on bars; muted axis."""
    cd = CategoryChartData(); cd.categories = cats
    for name, vals in series: cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, x, y, w, h, cd); ch = gf.chart
    ink = DARK_INK if dark else INK; ink2 = RGBColor(0xB2,0xB8,0xC1) if dark else INK2; grid = RGBColor(0x2C,0x32,0x3A) if dark else RULE
    ch.has_title = False; ch.has_legend = bool(legend) and len(series) > 1
    if ch.has_legend:
        ch.legend.position = legend; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size); ch.legend.font.name = SANS; ch.legend.font.color.rgb = ink2
    ch.font.name = SANS; ch.font.size = Pt(size); ch.font.color.rgb = ink2
    ca, va = ch.category_axis, ch.value_axis
    ca.tick_labels.font.size = Pt(size); ca.tick_labels.font.color.rgb = ink; ca.format.line.color.rgb = grid
    ca.has_major_gridlines = False; ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    if horizontal: ca.reverse_order = True
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = grid; va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1); va.tick_labels.font.color.rgb = ink2; va.tick_labels.number_format = fmt; va.tick_labels.number_format_is_linked = False
    if axis_max is not None: va.maximum_scale = axis_max
    if axis_min is not None: va.minimum_scale = axis_min
    plot = ch.plots[0]; plot.gap_width = gap; plot.overlap = overlap; plot.vary_by_categories = False
    plot.has_data_labels = True; dl = plot.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.font.size = Pt(size - 1); dl.font.color.rgb = ink; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    pal = colors or [DATA, ACCENT, INK3]
    for i, ser in enumerate(plot.series):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = pal[i % len(pal)]; ser.format.line.fill.background()
    return ch

def point_colors(chart, colors):
    """Colour individual bars of the first series (e.g. null rows in grey)."""
    ser = chart.plots[0].series[0]
    for i, c in enumerate(colors):
        if c is None: continue
        pt = ser.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c

n = 0
def new(dark=False):
    global n; n += 1; s = prs.slides.add_slide(BLANK); bg(s, DARK if dark else PAPER); footer(s, n, dark); return s

# ================= 1. TITLE =================
s = new(dark=True)
text(s, M, Inches(2.0), Inches(11), Inches(0.4), "PAPER TALK  ·  SEPTEMBER 2026", size=12, color=DARK_ACC, bold=True)
text(s, M, Inches(2.4), Inches(11.5), Inches(1.6), "Talking at the Audience", size=56, font=SERIF, color=DARK_INK)
text(s, M, Inches(3.75), Inches(11), Inches(1.2), "Listener-directed address in political podcasts, and the institutional cynicism of the people who listen",
     size=22, font=SERIF, color=RGBColor(0xB2, 0xB8, 0xC1))
text(s, M, Inches(5.4), Inches(11), Inches(0.9), [[("Danielle Yang", {"bold": True, "color": DARK_INK}), ("   with R. Michael Alvarez", {"color": RGBColor(0xB2, 0xB8, 0xC1)})],
     [("Linde Center for Science, Society, and Policy · Caltech", {"color": RGBColor(0x7F, 0x87, 0x92), "size": 14})]], size=16)
notes(s, "Open with the one-sentence version: political podcasts are a huge news source we have barely measured; left and right differ in how hosts talk TO listeners, not in how hostile they are; and podcast audiences are distinctively cynical about institutions. Everything else in the talk is evidence for those three claims.")

# ================= 2. MOTIVATION =================
s = new(); kicker(s, "Motivation"); title(s, "A major political medium, mostly unmeasured")
bullets(s, M, Inches(2.2), Inches(6.6), Inches(4.2), [
    "Podcasts are now a primary political news source for millions of Americans, and the largest political shows out-draw cable news.",
    "Almost all of what we know about partisan media comes from television and print: incivility, outrage, strategy framing.",
    "Podcasts are different in form: long, intimate, one voice speaking directly to a listener for hours a week.",
    "The common assumption is that the right-leaning shows are angrier. Nobody had measured it at scale.",
    "And nobody had linked what hosts say to who is actually listening and what those listeners believe.",
], size=17)
box(s, Inches(7.9), Inches(2.2), Inches(4.6), Inches(3.9), fill=RAISED, line=RULE)
text(s, Inches(8.2), Inches(2.4), Inches(4.0), Inches(0.4), "TWO GAPS THIS PAPER FILLS", size=11, color=ACCENT, bold=True)
text(s, Inches(8.2), Inches(2.85), Inches(4.0), Inches(3.2), [
    [("1  Content. ", {"bold": True}), ("A transcribed, validated corpus of the political podcast field, with register measured on it.", {})],
    [("2  Audience. ", {"bold": True}), ("The corpus linked to two national probability surveys, so shows can be matched to their own listeners.", {})],
], size=15, color=INK2, space_after=14)
notes(s, "Set up the two gaps. Gap one is content: we did not have a corpus. Gap two is audience: even where people have studied podcast content, nobody could connect a specific show's language to its own listeners. The survey linkage is what makes this paper different.")

# ================= 3. RESEARCH QUESTIONS =================
s = new(); kicker(s, "Research questions"); title(s, "Three questions")
qs = [("RQ1", "How do left- and right-leaning political podcasts differ in how hosts talk?", "Hostility? Moral language? Or something about the relationship the host sets up with the listener?"),
      ("RQ2", "Who listens to political podcasts, and how do they differ from other news consumers?", "Institutional trust, democratic norms, partisanship, participation — measured in national surveys."),
      ("RQ3", "Does a show's mode of address predict who its audience is and what they believe?", "The show-level link: register measured on transcripts, outcomes measured on the show's own listeners.")]
for i, (tag, q, sub) in enumerate(qs):
    y = Inches(2.2) + i * Inches(1.5)
    text(s, M, y, Inches(1.0), Inches(0.6), tag, size=22, font=SERIF, color=ACCENT)
    text(s, M + Inches(1.1), y - Inches(0.02), Inches(10.5), Inches(0.6), q, size=20, font=SERIF, color=INK)
    text(s, M + Inches(1.1), y + Inches(0.6), Inches(10.5), Inches(0.6), sub, size=14, color=INK2)
notes(s, "RQ1 is descriptive and the measurement contribution. RQ2 is the population finding and does not depend on the corpus at all. RQ3 is where the two meet, and it is the part that is exploratory — say that now so it does not surprise anyone at slide 18.")

# ================= 4. LITERATURE =================
s = new(); kicker(s, "Where it sits"); title(s, "Three literatures, one prediction each")
rows = [["Literature", "Anchor", "What it predicts here", "What we find"],
        ["Media malaise / spiral of cynicism", "Robinson 1976; Cappella & Jamieson 1997; Ladd 2012", "News diets outside institutional journalism go with lower institutional trust", "Supported (RQ2)"],
        ["Incivility and outrage", "Mutz & Reeves 2005; Sobieraj & Berry 2011", "Right-leaning shows are more hostile; hostility drives distrust", "Not supported for the partisan contrast"],
        ["Para-social interaction", "Horton & Wohl 1956", "Direct personal address builds a one-sided relationship with the performer", "Consistent: address, not hostility, is the partisan difference"],
        ["Register variation", "Biber 1988", "Involved vs. informational style is a stable property of a speaker", "Address is a stable show trait; reaches less-educated listeners"]]
table(s, M, Inches(2.2), W - 2 * M, rows, col_w=[Inches(2.6), Inches(3.0), Inches(3.6), Inches(2.5)], size=12, row_h=Inches(0.62))
text(s, M, Inches(5.6), W - 2 * M, Inches(0.8), "Populist communication (Rooduijn & Pauwels 2011; Fawzi 2019) enters as a rival content feature, carried as a covariate rather than ruled out.", size=13, color=INK2)
notes(s, "Keep this to 60 seconds. The point is that the outrage literature makes a clear prediction about podcasts and the data do not support it, while the parasocial and register literatures fit what we see.")

# ================= 5. DATA: CORPUS =================
s = new(); kicker(s, "Data"); title(s, "The corpus")
stat_row(s, Inches(2.1), [("204", "political podcasts from the Apple Podcasts US Politics chart, ranks 1–250, frozen 13 July 2026"),
                          ("29,421", "episodes transcribed (faster-whisper large-v3-turbo, two GB10 nodes, ~100× real time)"),
                          ("1.38 M", "750-character passages, the unit the ideology instrument was validated on"),
                          ("2018–2026", "with a census of the pre-2018 tail; episodes drawn per show per quarter so show-time is aligned")])
bullets(s, M, Inches(3.8), Inches(11.7), Inches(2.5), [
    "Inclusion filter applied in fixed order: 250 charted → 219 (non-US-politics, under 10 hours of audio, no usable feed) → 204 on the topic-arm sample.",
    "Scope: political podcasts that attained and held Politics-chart presence. Mid-tail by construction; the largest political shows (Shapiro, The Daily, Tucker) are filed under other categories and enter as an out-of-frame test.",
    "Ideology labelled at passage level by a two-stage LLM classifier (152,152 passages), aggregated to show level; host ideology anchored externally with DIME campaign-contribution scores (112 shows).",
], size=15)
notes(s, "If asked about the 204 vs 219: fifteen shows were not in the transcript sample when the topic arm was fixed; they skew to the chart tail and legacy outlets, and we state that as a scope condition. Median episode is 47 minutes.")

# ================= 6. DATA: SURVEYS =================
s = new(); kicker(s, "Data"); title(s, "Two national surveys, linked to the corpus")
rows = [["Survey", "Sample", "Exposure", "Outcomes"],
        ["Kettering Foundation / Gallup, Democracy for All, Year 1 (Jul–Aug 2025)", "20,338; 45% Gallup Panel (probability), 55% opt-in; weighted",
         "Open-ended top-three news sources: 51,082 verbatims. Matched to shows by exact title (385 respondents), plus a re-scan for misspellings (150), plus six out-of-frame political podcasts (290)",
         "8-item institutional cynicism (α = .84), election distrust, democratic-norms battery, participation"],
        ["Pew American Trends Panel, Waves 150 / 155 / 165 (Jul 2024 – Mar 2025)", "10,658 in W150; ~9,000 linked to each later wave; probability panel",
         "Frequency of podcast news; podcasts as main election-news source; seven items on relationships with news influencers",
         "Trust in national news, local news, social media, and friends/family — measured 2–8 months after exposure"]]
table(s, M, Inches(2.2), W - 2 * M, rows, col_w=[Inches(2.9), Inches(2.6), Inches(3.6), Inches(2.6)], size=12, row_h=Inches(1.15))
claim(s, Inches(5.5), "Kettering lets us match a show's transcripts to that show's own listeners. Pew lets us put exposure before outcome.")
notes(s, "The Kettering open-ended item is the key asset: people typed the names of shows. That is how 792 listeners across 56 political podcasts get linked to register scores. Pew has no show names but has time order.")

# ================= 7. METHODS: REGISTER =================
s = new(); kicker(s, "Methods"); title(s, "Measuring how a host talks to the listener")
bullets(s, M, Inches(2.2), Inches(6.4), Inches(4.2), [
    [("Listener-directed address ", {"bold": True}), ("= mean of three show-level z-scores, each an n-weighted rate per 10,000 words:", {})],
    (">", "syntactic imperatives — a bare-form verb heading a clause with no subject or auxiliary (spaCy dependency parse)"),
    (">", "dictionary imperatives — look, listen, think about it, remember, understand …"),
    (">", "second-person pronouns — you, your, yourself"),
    [("Hostility ", {"bold": True}), ("from Sobieraj & Berry's outrage categories plus a profanity list; moral language from eMFD and MFD 2.0; Biber dimensions via the Multi-Feature Tagger of English.", {})],
    "All instruments published before this study. No word list was written for it.",
    "Register scored on every passage (ads flagged, not removed); topic model run on ad-excised text.",
], size=15)
box(s, Inches(7.7), Inches(2.2), Inches(4.8), Inches(4.0), fill=RAISED, line=RULE)
text(s, Inches(8.0), Inches(2.4), Inches(4.3), Inches(0.4), "WHAT THE MEASURE PICKS UP", size=11, color=ACCENT, bold=True)
text(s, Inches(8.0), Inches(2.85), Inches(4.3), Inches(3.3), [
    [("“You need to understand what they’re doing here. Look at what happened last week. Don’t let them tell you otherwise.”", {"italic": True, "font": SERIF, "size": 15})],
    [("Deontic second person (you need to, you should) is the strongest single component against host ideology (r = +0.44). The filler ‘you know’ is null. Biber’s persuasion dimension is uncorrelated (r = −0.12). The second-person subject carries the signal, not the modality.", {"size": 13, "color": INK2})],
], size=14, space_after=12)
notes(s, "Read the example aloud. The measure is about the grammatical relationship between host and listener, not about topic or tone. Mention the boundary: in interview shows 'you' is often addressed to the guest; the corpus is monologue-dominated and we carry a format covariate where it matters.")

# ================= 8. METHODS: LABELS =================
s = new(); kicker(s, "Methods"); title(s, "Ideology labels validated four ways")
rows = [["Standard", "Basis", "n", "Agreement"],
        ["DIME cfscore", "hosts' campaign contributions", "112 shows", "r = +0.68, κ = 0.66"],
        ["Editorial ratings", "human-assigned partisan leaning", "25", "κ = 0.75"],
        ["Expert ratings", "independent research coding", "26", "κ = 0.77"],
        ["Audience party ID", "party of the show's own Kettering listeners", "22 shows / 494 listeners", "r = +0.91 [0.77, 0.98]"]]
table(s, M, Inches(2.2), Inches(8.2), rows, col_w=[Inches(2.0), Inches(3.0), Inches(1.6), Inches(1.6)], size=13, hi_cols=(3,), row_h=Inches(0.45))
text(s, Inches(9.4), Inches(2.2), Inches(3.2), Inches(3.5), [
    [("21 of 22", {"size": 40, "font": SERIF, "color": DATA})],
    [("shows whose label predicts the majority party of their own listeners", {"size": 13, "color": INK2})],
    [("Left-labelled shows average 76% Democratic audiences; right-labelled 91% Republican.", {"size": 13, "color": INK2})],
], size=14, space_after=10)
text(s, M, Inches(4.9), Inches(11.5), Inches(1.0), "Text, money and listeners agree. This validates show labels; a 200-passage two-coder check of passage labels precedes submission.", size=15, color=INK2)
notes(s, "The audience validation is the one reviewers have not seen before: the label predicts the party of people who typed the show's name into a survey. That is independent of the model and of the text.")

# ================= 9. METHODS: INFERENCE =================
s = new(); kicker(s, "Methods"); title(s, "Inference built for few, unequal clusters")
bullets(s, M, Inches(2.2), Inches(11.7), Inches(4.5), [
    [("Between-show register differences: ", {"bold": True}), ("show-level permutation tests, Benjamini–Hochberg across measure families, format and ad controls, a discriminant null (filler ‘you know’), and equivalence (TOST) tests for the nulls.", {})],
    [("Population survey estimates: ", {"bold": True}), ("weighted OLS with robust SEs, full control ladder (party, attention, age, education, gender, race, income, social-media use, platforms, urbanicity, mode, number of sources), within-party and within-education re-estimation.", {})],
    [("Show-linked audience estimates: ", {"bold": True}), ("exposure is assigned by show and a few shows hold most listeners, so conventional clustered SEs are anti-conservative. Every estimate reports CR2 (Bell–McCaffrey) with Satterthwaite df as the primary p and a wild cluster bootstrap alongside; the df (≈ 9 on the full frame) is the honest cluster count.", {})],
    [("Pre-registration: ", {"bold": True}), ("the out-of-frame test was frozen and hashed before any audio was fetched — hypothesis, alias table, outcome, model, inference, failure criterion.", {})],
], size=15, gap=12)
notes(s, "This slide exists so the methods-minded people in the room relax. The key phrase is Satterthwaite degrees of freedom: with 56 clusters and MeidasTouch holding 134 listeners, the effective number of clusters is about nine, and we report it.")

# ================= 10. RESULT 1: ADDRESS NOT HOSTILITY =================
s = new(dark=True); kicker(s, "Results", dark=True); title(s, "Finding 1 — Left and right differ in mode of address, not hostility", dark=True, size=30)
cats = ["Listener-directed address (composite)", "Imperatives (syntactic)", "Second-person address", "Out-group pronouns", "Insult", "Hedging", "Profanity"]
vals = [0.43, 0.39, 0.39, 0.29, 0.00, -0.13, -0.17]
ch = bar_chart(s, M, Inches(2.35), Inches(7.4), Inches(3.9), cats, [("r with host ideology (DIME, n = 112)", vals)], dark=True, colors=[DARK_ACC], size=12, axis_min=-0.3, axis_max=0.6)
point_colors(ch, [None, None, None, None, INK3, INK3, INK3])
text(s, M, Inches(6.3), Inches(7.4), Inches(0.4), "Pearson r between each show-level register measure and the host's DIME contribution score. Grey bars: not significant.", size=11, color=RGBColor(0xB2, 0xB8, 0xC1))
box(s, Inches(8.6), Inches(2.35), Inches(4.0), Inches(4.2), fill=RGBColor(0x1B,0x1F,0x25), line=RGBColor(0x2C,0x32,0x3A))
text(s, Inches(8.9), Inches(2.55), Inches(3.5), Inches(3.9), [
    [("d = 0.78", {"size": 40, "font": SERIF, "color": DARK_ACC})],
    [("right vs left shows on listener-directed address, format-controlled (p < .0001)", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
    [("Insult: r = 0.00", {"size": 24, "font": SERIF, "color": DARK_INK})],
    [("tracks ideological intensity (r = +0.55), not direction", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
    [("AUC 0.80 / 0.57", {"size": 24, "font": SERIF, "color": DARK_INK})],
    [("a register classifier separates left from right between shows, barely within them: register is a property of the show", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
], size=14, space_after=8)
notes(s, "This is the headline for RQ1. The chart is correlation with host donations: the address measures sit at .3 to .4, insult at exactly zero. Nine of ten register differences survive the format control (right-leaning shows are 84% solo-hosted vs 58% on the left). Full table with audience-lean correlations and between-side d values is in the backup deck / paper.")

# ================= 11. RESULT 1b: STABLE TRAIT =================
s = new(); kicker(s, "Results"); title(s, "Finding 1, continued — It is how they talk, not what is happening")
bullets(s, M, Inches(2.2), Inches(6.5), Inches(4.2), [
    [("Eight political shocks, 2020–2025: ", {"bold": True}), ("two general elections, a midterm, the inauguration, the assassination attempt, Biden’s withdrawal, January 6, Dobbs.", {})],
    "Within-show event study: listener-directed address does not shift on either side (all |b| ≤ 0.08 SD).",
    "The 2024 election ‘shift’ is indistinguishable from 60 placebo dates (p = .57).",
    "Across a 75-topic model, within-show topic means spread only 0.04 SD; the most second-person passages are ad reads.",
    "Hosts do not ramp it up before elections and do not change it by subject. It is a trait of the host.",
], size=16)
box(s, Inches(7.8), Inches(2.2), Inches(4.7), Inches(3.6), fill=RAISED, line=RULE)
text(s, Inches(8.1), Inches(2.4), Inches(4.2), Inches(0.4), "WHY THIS MATTERS", size=11, color=ACCENT, bold=True)
text(s, Inches(8.1), Inches(2.85), Inches(4.2), Inches(2.9), "If address were a reaction to the news cycle, it would be a poor candidate for a stable audience-selection mechanism. Because it is invariant to events and topics, a listener who chooses a show is choosing a mode of address that will be there every week.", size=14, color=INK2)
notes(s, "Keep short. The event study and the topic check are both robustness for the claim that the measure is a show trait, which is what licenses using it as a show-level exposure later.")

# ================= 12. RESULT 2: POPULATION =================
s = new(dark=True); kicker(s, "Results", dark=True); title(s, "Finding 2 — Podcast news diets go with institutional cynicism", dark=True, size=30)
cats = ["Podcast-only diet (4.3% of adults)", "Platform-only diet (16.6%)", "Podcast + mainstream (5.5%)", "Any podcast named vs other news consumers", "   within Democrats & leaners", "   within Republicans & leaners", "   low education", "   high education"]
vals = [0.269, 0.178, 0.132, 0.218, 0.188, 0.152, 0.34, 0.35]
ch = bar_chart(s, M, Inches(2.35), Inches(7.6), Inches(4.0), cats, [("institutional cynicism, SD units", vals)], dark=True, colors=[DARK_ACC], size=12, axis_min=0, axis_max=0.45)
point_colors(ch, [None, INK3, INK3, DATA, DATA, DATA, DATA, DATA])
text(s, M, Inches(6.4), Inches(7.6), Inches(0.4), "Kettering Y1, 8-item cynicism vs mainstream-only (first three) or vs all other news consumers; weighted; full control ladder. All p < .001.", size=11, color=RGBColor(0xB2, 0xB8, 0xC1))
box(s, Inches(8.8), Inches(2.35), Inches(3.8), Inches(4.2), fill=RGBColor(0x1B,0x1F,0x25), line=RGBColor(0x2C,0x32,0x3A))
text(s, Inches(9.1), Inches(2.55), Inches(3.3), Inches(3.9), [
    [("−0.10 SD", {"size": 34, "font": SERIF, "color": DARK_ACC})],
    [("for each additional mainstream source named (p = 2e-21); +0.08 for each additional podcast", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
    [("+0.14 SD", {"size": 34, "font": SERIF, "color": DARK_INK})],
    [("podcast-only beyond platform-only; equivalence rejected at ±0.20", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
    [("n = 18,257 · coefficient moves 3% across the control ladder · identical at every education level (interaction p = .91)", {"size": 12, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
], size=14, space_after=8)
notes(s, "This is the result that does not depend on the corpus and is the strongest thing in the paper. Read the chart top to bottom: diets with no institutional source are about 0.2 SD more cynical; podcast-only is the far end. Then the within-group bars: same in both parties, same at both education levels. The gradient in the side panel is the mechanism-shaped fact: every mainstream source you name is worth −0.10 SD.")

# ================= 13. RESULT 2b: PEW =================
s = new(); kicker(s, "Results"); title(s, "Finding 2, continued — Replicates in Pew with exposure measured first")
cats = ["national news organisations", "local news organisations", "social media sites", "friends, family, acquaintances"]
ch = bar_chart(s, M, Inches(2.2), Inches(7.4), Inches(3.9), cats, [("W155, Sept 2024", [-0.066, -0.067, 0.140, 0.011]), ("W165, March 2025", [-0.083, -0.083, 0.154, 0.024])],
               horizontal=True, colors=[DATA, ACCENT], legend=XL_LEGEND_POSITION.BOTTOM, size=12, fmt='+0.000;-0.000;0.000', axis_min=-0.12, axis_max=0.2, gap=60, overlap=-5)
text(s, M, Inches(6.15), Inches(7.4), Inches(0.5), "Trust, SD per step of podcast-news frequency measured July–Aug 2024 (1 never – 4 often); n ≈ 9,000 per wave; weighted, controls. First three bars p < .0001; friends and family ns.", size=11, color=INK3)
box(s, Inches(8.6), Inches(2.2), Inches(4.0), Inches(4.2), fill=RAISED, line=RULE)
text(s, Inches(8.9), Inches(2.4), Inches(3.5), Inches(3.9), [
    [("−0.34 SD", {"size": 34, "font": SERIF, "color": ACCENT})],
    [("trust in national news, March 2025, among those who said podcasts were their main election-news source", {"size": 13, "color": INK2})],
    [("The shape is the point:", {"size": 16, "bold": True})],
    [("institutional trust falls, trust in the informal channel rises, interpersonal trust is flat. A response-style artifact cannot produce that pattern. The stricter exposure gives effects three to four times larger.", {"size": 13, "color": INK2})],
], size=14, space_after=8)
notes(s, "Temporal order, not identification — say that. The friends-and-family bars are the discriminant check: if podcast listeners were just generally distrustful people, those bars would drop too.")

# ================= 14. RESULT 2c: WHO THEY ARE =================
s = new(); kicker(s, "Results"); title(s, "Finding 2, continued — Engaged anti-institutionalism, not disengagement")
cats = ["Media should take direction from government", "Expand presidential power", "Bar radicals from office", "Radicals should be able to protest", "Strong partisan identification"]
ch = bar_chart(s, M, Inches(2.2), Inches(7.2), Inches(3.7), cats, [("Podcast-only vs mainstream-only", [-0.35, -0.17, -0.18, 0.17, 0.00]), ("Platform-only vs mainstream-only", [0.0, 0.0, 0.0, 0.0, 0.0])],
               colors=[ACCENT, INK3], legend=XL_LEGEND_POSITION.BOTTOM, size=12, axis_min=-0.45, axis_max=0.3, gap=70, overlap=-10)
text(s, M, Inches(5.95), Inches(7.2), Inches(0.6), "Democratic-norms battery, SD units, full controls, BH-corrected across all 115 items. Platform-only shows none of the civil-libertarian pattern (bars at zero).", size=11, color=INK3)
rows = [["Where the two non-institutional diets diverge", "Podcast-only", "Platform-only"],
        ["Registered to vote", "no difference", "lower"], ["Tolerance of political violence", "no difference", "higher"],
        ["National identity", "no difference", "weaker"], ["Loneliness", "no difference", "—"]]
table(s, Inches(8.4), Inches(2.2), Inches(4.2), rows, col_w=[Inches(2.2), Inches(1.0), Inches(1.0)], size=11, row_h=Inches(0.42))
text(s, Inches(8.4), Inches(4.6), Inches(4.2), Inches(1.8), "Podcast-only listeners distrust institutions and are more protective of civil liberties, no more partisan, no lonelier, no less registered. Platform-only diets carry weaker national identity, income strain, lower registration and more tolerance of political violence.", size=13, color=INK2)
notes(s, "This is the slide that stops people reading finding 2 as 'podcasts make people cynical and checked out'. The podcast-only profile is cynical AND civil-libertarian AND engaged. The platform-only profile is the disengaged one.")

# ================= 15. RESULT 3: EDUCATION =================
s = new(dark=True); kicker(s, "Results", dark=True); title(s, "Finding 3 — Listener-directed shows reach less formally educated audiences", dark=True, size=30)
cats = ["Corpus frame (54 shows, n = 535)", "  discovery listeners only (51 / 385)", "  fresh corpus listeners only (10 / 150)", "Out-of-frame political podcasts (6 / 290)", "Full political-podcast frame (56 / 792)"]
ch = bar_chart(s, M, Inches(2.35), Inches(7.4), Inches(3.6), cats, [("audience education, SD per SD of address", [-0.272, -0.285, -0.160, -0.082, -0.213])], dark=True, colors=[DARK_ACC], size=12, axis_min=-0.35, axis_max=0)
point_colors(ch, [None, DATA, DATA, DATA, None])
text(s, M, Inches(6.05), Inches(7.4), Inches(0.5), "Party and show lean controlled; CR2 and wild-bootstrap p ≤ .048 in every sample. Unchanged with age, gender, income, race, urbanicity and attention added (−0.28).", size=11, color=RGBColor(0xB2, 0xB8, 0xC1))
box(s, Inches(8.6), Inches(2.35), Inches(4.0), Inches(4.2), fill=RGBColor(0x1B,0x1F,0x25), line=RGBColor(0x2C,0x32,0x3A))
text(s, Inches(8.9), Inches(2.55), Inches(3.5), Inches(3.9), [
    [("4 of 4", {"size": 40, "font": SERIF, "color": DARK_ACC})],
    [("samples replicate, including the out-of-frame shows where nothing else did", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
    [("Education specifically", {"size": 18, "font": SERIF, "color": DARK_INK})],
    [("income, age and gender are null; holds within left shows (−0.29) and right shows (−0.19); stable to dropping any of the 12 largest shows; show-level r = −0.57", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
    [("First-source namers −0.35 vs later −0.20: the fingerprint of selection.", {"size": 13, "color": RGBColor(0xB2, 0xB8, 0xC1)})],
], size=14, space_after=8)
notes(s, "This is the most robust thing address predicts: it replicates in four samples including the out-of-frame shows. Frame it as register–audience fit: direct, imperative, second-person speech is the involved oral register, and it reaches people with less schooling regardless of party. It is about who a style reaches, not what it does to them.")

# ================= 16. RESULT 4: PARASOCIAL =================
s = new(); kicker(s, "Results"); title(s, "Finding 4 — Podcast news use carries the para-social signature")
cats = ["feel a personal connection to an influencer", "follow or subscribe", "influencers helped me understand", "get opinions and takes from them", "their opinions mostly agree with mine", "get basic facts from them", "their news is different from other news"]
ch = bar_chart(s, M, Inches(2.2), Inches(7.4), Inches(3.9), cats, [("percentage points per step of podcast-news frequency", [4.1, 6.0, 5.8, 3.5, 2.6, 1.2, 0.0])], colors=[ACCENT], size=12, fmt='+0.0;-0.0;0.0', axis_min=0, axis_max=7)
point_colors(ch, [None, None, None, None, None, INK3, INK3])
text(s, M, Inches(6.15), Inches(7.4), Inches(0.5), "Pew W150, regular consumers of news from influencers (n = 2,012); BH-corrected. Grey: not significant (q = .19, .69).", size=11, color=INK3)
box(s, Inches(8.6), Inches(2.2), Inches(4.0), Inches(4.2), fill=RAISED, line=RULE)
text(s, Inches(8.9), Inches(2.4), Inches(3.5), Inches(3.9), [
    [("26% → 39%", {"size": 34, "font": SERIF, "color": ACCENT})],
    [("share who feel a personal connection to an influencer, from never to often getting news from podcasts; holds within both parties", {"size": 13, "color": INK2})],
    [("The relational and opinion items move; the informational items do not.", {"size": 15, "bold": True})],
    [("That is the audience-side signature of direct, personal address.", {"size": 13, "color": INK2})],
], size=14, space_after=8)
notes(s, "Bridge slide: the register we measure on the host side has a matching signature on the listener side, in an independent survey. Keep it to 45 seconds.")

# ================= 17. RESULT 5: ADDRESS -> DISTRUST =================
s = new(dark=True); kicker(s, "Results · exploratory", dark=True); title(s, "Finding 5 — Within political podcasts, address goes with election distrust", dark=True, size=30)
cats = ["Election distrust", "Institutional cynicism (8 items)", "Disaffection", "Economic grievance", "Exclusivity"]
ch = bar_chart(s, M, Inches(2.35), Inches(7.6), Inches(3.4), cats, [("corpus shows (50)", [0.22, 0.24, 0.17, 0.16, 0.26]), ("out-of-frame political podcasts (6)", [0.28, 0.23, 0.27, 0.19, 0.04]), ("pooled, net of populism + intensity", [0.19, 0.16, 0.14, 0.13, 0.02])],
               horizontal=True, dark=True, colors=[DATA, DARK_ACC, RGBColor(0x7F,0x87,0x92)], legend=XL_LEGEND_POSITION.BOTTOM, size=11, axis_min=0, axis_max=0.35, gap=50, overlap=-5)
rows = [["Pooled (56 shows, 792 listeners)", "b / SD", "CR2 / wild p"], ["Election distrust", "+0.267", "< .001 / < .001"], ["Institutional cynicism", "+0.221", ".005 / < .001"], ["Disaffection", "+0.159", "< .001 / .003"], ["Economic grievance", "+0.182", ".007 / .005"], ["Exclusivity", "+0.089", ".066 / .055"]]
table(s, Inches(8.8), Inches(2.35), Inches(3.8), rows, col_w=[Inches(1.9), Inches(0.8), Inches(1.1)], size=11, hi_cols=(1,), row_h=Inches(0.4), dark=True)
text(s, Inches(8.8), Inches(4.9), Inches(3.8), Inches(1.6), "Political podcast = at least 15% of passages political (corpus 10th percentile), applied to corpus and out-of-frame shows alike. Election distrust stays +0.20 to +0.29 at every cutoff from 8% to 30% and survives dropping any show; the six out-of-frame shows carry only 1.7 effective df alone.", size=11, color=RGBColor(0xB2, 0xB8, 0xC1))
text(s, M, Inches(5.85), Inches(7.6), Inches(0.8), "SD per SD of listener-directed address; full controls (party, attention, age, education, gender, race, income, urbanicity, right-wing platforms, show lean, frame). Election distrust and cynicism are the two associations that survive populist vocabulary and ideological intensity in the same model.", size=11, color=RGBColor(0xB2, 0xB8, 0xC1))
notes(s, "Say 'exploratory' out loud before the numbers. The chart shows the same pattern inside the corpus and among the largest political podcasts outside it, and the grey bars show what remains once the two rival content features are in the model. The out-of-frame test was pre-registered on a chart-based set of ten shows and did not pass there; two of the ten turned out to be non-political by content. The content definition came afterward. Do not oversell this slide; the next one is where the honesty lives.")

# ================= 18. PRE-REGISTRATION =================
s = new(); kicker(s, "Design"); title(s, "What was pre-registered, and what happened")
rows = [["Test", "Frozen", "Outcome"],
        ["Out-of-frame generalization: address → cynicism among 749 fresh listeners of ten large chart-absent shows", "2 Sept 2026, hashed, before any audio was fetched; two amendments filed before any outcome was opened", "Did not pass (b = −0.001; Rogan-excluded +0.22, p = .08). Reported as a result."],
        ["Within-frame fresh-listener test: 150 corpus-show listeners recovered by re-scan", "3 Sept 2026, before outcomes; Majority Report scored blind", "Did not pass (b = −0.06)."],
        ["Q28 free-text test (‘what does democracy mean to you’)", "Timestamped before the variable was opened", "Did not pass (p = .125)."]]
table(s, M, Inches(2.2), W - 2 * M, rows, col_w=[Inches(4.6), Inches(4.0), Inches(3.1)], size=12, row_h=Inches(0.8))
text(s, M, Inches(5.2), W - 2 * M, Inches(1.5), [
    [("Everything after those tests is labelled exploratory: ", {"bold": True}), ("the political-content frame, the education-composition result, and the comparison with populist vocabulary and ideological intensity were found by searching and are reported with the search disclosed (an outcome-scan family table is in the methods).", {})],
    [("The three content features inter-correlate at 0.4–0.5; with about 56 clusters we do not claim to have adjudicated among them beyond finding 5.", {"color": INK2})],
], size=15, space_after=10)
notes(s, "This slide is deliberate. Reviewers at Political Communication will find the hashed pre-registration; it is far better that the room hears it from us. The framing: hypothesis generated in discovery, tested, not confirmed on the frozen frame, recovered on a content-defined frame, pre-registered for Year 2.")

# ================= 19. LIMITATIONS =================
s = new(); kicker(s, "Limits"); title(s, "Scope and limitations")
bullets(s, M, Inches(2.2), Inches(11.7), Inches(4.6), [
    [("Association, not effect. ", {"bold": True}), ("People choose their podcasts. The Pew lag gives temporal order, not identification. No dose–response on any attitude; the education gradient sharpens with dose, which is what selection predicts.", {})],
    [("Scope. ", {"bold": True}), ("Politics-chart shows plus the six largest political podcasts outside it; population defined by content. The content definition is post hoc and pre-registered for Year 2.", {})],
    [("Address is format-sensitive. ", {"bold": True}), ("In interview shows the second person is often addressed to the guest. The partisan contrast survives removing the interrogative component; cross-format levels need diarization, which the archived audio permits.", {})],
    [("Content features are collinear. ", {"bold": True}), ("Address, populist vocabulary and ideological intensity co-vary across shows and are jointly associated with audience disaffection; Year 1 ranks them only within the political frame.", {})],
    [("Validation gaps. ", {"bold": True}), ("Show labels validated four ways; passage labels not yet human-coded. Ad-excision gold set annotated by an LLM, human re-annotation pending.", {})],
    [("Small effective cluster counts, ", {"bold": True}), ("reported honestly through the Satterthwaite df.", {})],
], size=15, gap=10)
notes(s, "Read these as strengths of the write-up, not apologies. Each has a concrete next step on the following slide.")

# ================= 20. TAKEAWAYS =================
s = new(dark=True); kicker(s, "Takeaways", dark=True); title(s, "What the paper claims", dark=True)
for i, (h, b) in enumerate([
    ("How the sides differ", "Partisan difference in political podcasting lies in how hosts address their audience — second person and instructions — not in how hostile they are. It is a stable trait of the show, validated against donations, experts and listeners’ own party."),
    ("Who listens", "People who get political news from podcasts are about 0.22 SD more cynical about institutions than other news consumers, in two probability panels, one lagged, in both parties and at every education level — and they are civil-libertarian, not disengaged."),
    ("What address predicts", "Within political podcasts, the hosts who talk at the listener reach less-educated audiences (robust) who distrust elections and institutions (exploratory, pre-registered for Year 2)."),
]):
    y = Inches(2.2) + i * Inches(1.45)
    text(s, M, y, Inches(3.0), Inches(0.5), h, size=18, font=SERIF, color=DARK_ACC)
    text(s, M + Inches(3.2), y, Inches(8.6), Inches(1.3), b, size=15, color=DARK_INK)
notes(s, "Land the three claims in the same order as the opening. If there is one sentence to leave with: the register determines who listens; whether it also cultivates distrust is the question the next study is built to answer.")

# ================= 21. NEXT STEPS =================
s = new(); kicker(s, "Next"); title(s, "What comes next")
bullets(s, M, Inches(2.2), Inches(6.0), Inches(4.5), [
    [("Before submission", {"bold": True, "color": ACCENT})],
    (">", "200-passage, two-coder human check of ideology labels"),
    (">", "Human re-annotation of the 886-sentence ad gold set"),
    (">", "Disclosure paragraph and family table in the methods"),
    (">", "Freeze corpus counts; binary outcomes in percentage points"),
    [("Kettering Year 2 pre-registration", {"bold": True, "color": ACCENT})],
    (">", "Content-defined frame stated in advance"),
    (">", "Election distrust as the primary outcome for address"),
    (">", "Populist vocabulary and intensity as covariates; interview dates requested for within-show designs"),
], size=15, gap=6)
bullets(s, Inches(7.2), Inches(2.2), Inches(5.3), Inches(4.5), [
    [("Measurement", {"bold": True, "color": ACCENT})],
    (">", "Diarize archived audio for interview shows; rescore address on host turns only"),
    (">", "Band 6–10 corpus extension (in progress) for quarter-level analyses"),
    [("The causal study", {"bold": True, "color": ACCENT})],
    (">", "Survey experiment: the same segment with and without second-person and imperative address, randomly assigned; efficacy and election-trust items after"),
    (">", "The only design that answers whether address does anything to a listener rather than who it attracts"),
], size=15, gap=6)
notes(s, "If Alvarez is in the room, the experiment is the pitch. It is cheap, it uses the measure this paper validates, and it turns the exploratory finding into a testable causal claim.")

# ================= 22. THANKS =================
s = new(dark=True)
text(s, M, Inches(2.6), Inches(11), Inches(1.2), "Thank you", size=48, font=SERIF, color=DARK_INK)
text(s, M, Inches(3.8), Inches(11), Inches(1.4), [[("Danielle Yang · dsyang@caltech.edu", {"color": DARK_INK, "size": 18})],
     [("Linde Center for Science, Society, and Policy, Caltech · with R. Michael Alvarez", {"color": RGBColor(0xB2, 0xB8, 0xC1), "size": 15})],
     [("Corpus: 204 shows · 29,421 episodes · 1.38M passages · Kettering–Gallup Democracy for All Y1 · Pew ATP W150/155/165", {"color": RGBColor(0x7F, 0x87, 0x92), "size": 12})]], size=16, space_after=8)
notes(s, "Backup slides follow: sampling and pipeline details, inference detail, the full pre-registration table.")

# ================= BACKUP A: PIPELINE =================
s = new(); kicker(s, "Backup"); title(s, "Corpus construction and transcription, in detail")
bullets(s, M, Inches(2.2), Inches(11.7), Inches(4.6), [
    "Frame: Apple Podcasts US Politics chart, ranks 1–250, frozen 13 July 2026 (250 is as deep as Apple serves).",
    "Filter in fixed order: no feed URL (2), feed unreachable (2), unparseable (1), non-US-politics by blocklist/regex/language/curated list (14), under 10 hours of audio (12) → 219; 15 not in the topic-arm transcript sample → 204.",
    "Sampling: census of every episode before 2018, then a fixed number of episodes per show per quarter, ranked by a frozen hash so larger draws nest earlier ones.",
    "Transcription: faster-whisper batched pipeline, large-v3-turbo (CTranslate2, float16), batch 64, English fixed, Silero VAD, no diarization; two NVIDIA GB10 nodes at ~100× real time.",
    "Register arm: exhaustive 750-character chunks at sentence boundaries, nothing dropped; ads flagged. Topic arm: sentence-level ad/boilerplate excision (5.9% of words; gate ≥ .90 precision, ≤ 2% content false positives) then ~500-content-word passages.",
    "Ideology: two-stage gated LLM classifier at passage level; DIME cfscores for hosts via a three-stage matching cascade with logged manual resolution.",
], size=14, gap=8)
notes(s, "Only if asked.")

# ================= BACKUP B: INFERENCE DETAIL =================
s = new(); kicker(s, "Backup"); title(s, "Why CR2 and the wild bootstrap")
bullets(s, M, Inches(2.2), Inches(11.7), Inches(4.4), [
    "On the 56-show political-podcast frame, 57% of clusters hold fewer than five listeners and the four largest (MeidasTouch, Shapiro, NBC Nightly News, The Daily) hold 47%.",
    "Conventional CR1 standard errors are anti-conservative in that regime (Cameron, Gelbach & Miller 2008); a naive p of .0009 on one exclusivity estimate became .17 under the wild bootstrap.",
    "Every show-clustered estimate therefore reports CR2 (Bell–McCaffrey) with Satterthwaite degrees of freedom (Pustejovsky & Tipton 2018) as the primary p, and a Rademacher wild cluster bootstrap (1,000–2,000 replications) alongside. The two agree on every estimate in the paper.",
    "A show-level precision-weighted permutation of the exposure is reported as a low-power third check; leave-one-show-out and content-threshold sensitivity accompany every headline.",
    "Composites are standardized on the full Kettering sample, never on the listener subsample; address is standardized on the 194-show corpus reference, and out-of-frame shows are scored against that same reference.",
], size=15, gap=10)
notes(s, "Only if asked.")

out = str(__import__("pathlib").Path(__file__).resolve().with_name("Talking_at_the_Audience_slides.pptx"))  # next to this script
prs.save(out); print("saved", out, "| slides:", len(prs.slides))
