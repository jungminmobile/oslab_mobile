"""report_prelim_v9.md -> 대회 양식 docx (양식의 서식·표 모양을 그대로 따라 본문을 새로 만든다).
사용: python md2docx.py <md> <template.docx> <figs_dir> <out.docx> <teaminfo.json>
팀 정보(teaminfo.json: name, members[], contact, date)는 전화번호가 들어가므로 저장소 밖에 둔다. PDF 변환은 Word(COM)로 한다. 그림은 md의 ![캡션](파일) 줄로 넣는다.
"""
import json, os, re, shutil, struct, sys, zipfile
from xml.sax.saxutils import escape

MD, TPL, FIGS, OUT, TEAM = sys.argv[1:6]
team = json.load(open(TEAM, encoding="utf-8"))
md = open(MD, encoding="utf-8").read()
md = re.sub(r"<!--.*?-->", "", md, flags=re.S)

# ---------- 1장 팀 정보 교체 ----------
t1 = "\n".join([
    "| 구분 | 내용 |", "|---|---|",
    f"| 팀명 | {team['name']} |",
    "| 선택 주제 | 자유주제 — 자유 모델 최적화 |",
    "| 대상 모델명 | SigLIP2 ViT-B/32 (256×256) 이미지 인코더 (`timm/ViT-B-32-SigLIP2-256`) |",
] + [f"| {'팀원 (성명 / 학번 / 역할)' if i == 0 else ''} | {m} |" for i, m in enumerate(team["members"])] + [
    f"| 대표자 연락처 | {team['contact']} |",
    f"| 제출일 | {team['date']} |",
])
md = re.sub(r"(## 1\. 팀 정보\n\n)\|.*?\n\n", lambda m: m.group(1) + t1 + "\n\n", md, count=1, flags=re.S)
md = md.replace("[확인 필요: 팀명]", team["name"])
# 부록 B(그림 넣을 위치 메모)는 빼고 그림을 본문에 넣는다
md = re.sub(r"## 부록 B\..*?(?=## 부록 C\.)", "", md, flags=re.S)

FIG_AFTER = {  # 이 문자열로 시작하는 블록 뒤에 그림
    "- 블록 5부터 무너지고": ("F1_sensitivity.png", "그림 1. 민감도 지도 — 블록 하나만 INT8로 했을 때 FP32 대비 임베딩 cos (dev 6×200)"),
    "- **2×2 결과**": ("F2_ablation_2x2.png", "그림 2. 레지스터 캐시(M2)·회전의 2×2 단독 효과 (dev 전체, 고화질 Top-1 Δ)"),
    "- **가장 나쁜 층의 SNR 순서가": ("F4b_min_snr_vs_acc.png", "그림 3. 배포 눈금의 층별 8비트 출력 SNR(가장 나쁜 층·평균)과 dev 정확도 손실"),
    "| 모델 크기 (MB) | 378.36 | 95.65 | **−74.7** |": ("F3b_improvement_b32.png", "그림 4. 제출 모델 P2의 FP32 대비 개선율 (i7 노트북, FP32와 같은 실행에서 번갈아 측정)"),
    "- 지연은 Z·G·GF·ALL8F가": ("F5_hailo_b32.png", "그림 5. Hailo-10H 실칩 배치 1 지연과 dev 정확도 (같은 세션)"),
}
FIG_AFTER = {}  # v11부터는 md의 ![캡션](파일) 줄로 그림을 넣는다 (위 목록은 v9·v10용)
BOX = {"3.2", "3.3", "5.3", "6.2", "6.3", "6.6", "7.1", "9.1", "9.2", "9.3", "9.4"}
BOX -= set(filter(None, os.environ.get("REPORT_BOX_EXCLUDE", "").split(",")))  # v13: 쪽 넘김 때 상자가 통째로 넘어가 빈 공간이 생기는 절은 상자를 뺄 수 있다

# ---------- 블록 파싱 ----------
def parse(lines):
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip() or ln.strip() == "---":
            i += 1; continue
        if ln.startswith("```"):
            j = i + 1; buf = []
            while not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append(("code", buf)); i = j + 1; continue
        m = re.match(r"^!\[(.*)\]\((.*)\)\s*$", ln)
        if m:  # ![캡션](그림 파일), 캡션 끝에 "|70" 처럼 쓰면 본문 폭의 70%로 넣는다 (v13, 없으면 92%)
            cap, pct = m.group(1), None
            mm = re.match(r"^(.*)\|(\d{2})$", cap)
            if mm:
                cap, pct = mm.group(1), int(mm.group(2))
            out.append(("fig", (m.group(2), cap, pct))); i += 1; continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(("table", rows)); continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
        if m:
            items = []
            while i < len(lines):
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                if not m:
                    if lines[i].startswith("   ") and lines[i].strip() and items:
                        items[-1][2] += " " + lines[i].strip(); i += 1; continue
                    break
                lvl = 1 if len(m.group(1)) >= 2 else 0
                items.append([lvl, "num" if m.group(2)[0].isdigit() else "bul", m.group(3), int(m.group(2)[:-1]) if m.group(2)[0].isdigit() else 1])
                i += 1
            out.append(("list", items)); continue
        buf = [ln.strip()]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(\s*)([-*]|\d+\.)\s|^\||^```|^#|^!\[", lines[i]):
            buf.append(lines[i].strip()); i += 1
        out.append(("para", " ".join(buf)))
    return out

# ---------- XML 조각 ----------
def dispw(s):
    return sum(2 if ord(c) > 0x2E7F else 1 for c in re.sub(r"\*\*|`", "", s))

def runs(text, sz=None, base_b=False, color=None, font=None):
    """**굵게**, `코드`, [확인 필요…] 노란 강조."""
    xs = []
    parts = re.split(r"(`[^`]*`)", text)
    bold = base_b
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) >= 2:
            segs = [(p[1:-1], True)]
        else:
            segs = [(q, False) for q in p.split("**")]
        for k, (s, code) in enumerate(segs):
            if not code and k > 0:
                bold = not bold
            if not s:
                continue
            for piece in re.split(r"(\[확인 필요[^\]]*\])", s):
                if not piece:
                    continue
                rp = ""
                if code:
                    rp += '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                elif font:
                    rp += font
                if bold:
                    rp += "<w:b/><w:bCs/>"
                if color:
                    rp += f'<w:color w:val="{color}"/>'
                s2 = sz - 1 if (code and sz) else sz
                if s2:
                    rp += f'<w:sz w:val="{s2}"/><w:szCs w:val="{s2}"/>'
                if piece.startswith("[확인 필요"):
                    rp += '<w:highlight w:val="yellow"/>'
                xs.append(f'<w:r><w:rPr>{rp}</w:rPr><w:t xml:space="preserve">{escape(piece)}</w:t></w:r>')
    return "".join(xs)

def para(text, sz=None, jc=None, after=80, line=288, b=False, color=None, num=None, ind=None, keep=False):
    pp = ""
    if keep:
        pp += "<w:keepNext/>"
    if num:
        pp += f'<w:numPr><w:ilvl w:val="{num[1]}"/><w:numId w:val="{num[0]}"/></w:numPr>'
    pp += f'<w:spacing w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
    if ind:
        pp += ind
    if jc:
        pp += f'<w:jc w:val="{jc}"/>'
    return f"<w:p><w:pPr>{pp}</w:pPr>{runs(text, sz, b, color)}</w:p>"

def heading(text, lvl, pagebreak=False):
    if lvl == 1:
        pp = '<w:pStyle w:val="1"/>' + ("<w:pageBreakBefore/>" if pagebreak else "") + '<w:keepNext/><w:spacing w:before="300" w:after="120"/>'
        rp = '<w:b/><w:bCs/><w:color w:val="1A1A1A"/><w:sz w:val="24"/><w:szCs w:val="24"/>'
    else:
        pp = '<w:pStyle w:val="2"/><w:keepNext/><w:spacing w:before="200" w:after="90"/>'
        rp = '<w:b/><w:bCs/><w:color w:val="333333"/><w:sz w:val="21"/><w:szCs w:val="21"/>'
    return f'<w:p><w:pPr>{pp}</w:pPr><w:r><w:rPr>{rp}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'

BORDER = lambda c: "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="{c}"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
CELLMAR = '<w:tcMar><w:top w:w="70" w:type="dxa"/><w:left w:w="110" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/><w:right w:w="110" w:type="dxa"/></w:tcMar>'

def col_widths(rows, total):
    n = len(rows[0])
    if n == 2:
        first = 2340 if total >= 9000 else 2200
        return [first, total - first]
    sc = []
    for c in range(n):
        ws = [dispw(r[c]) for r in rows if c < len(r)]
        mx, mean = min(max(ws), 70), sum(ws) / len(ws)
        sc.append(max(8, 0.5 * mx + 0.5 * mean))
    mins = [min(1800, max(800, dispw(rows[0][c]) * 95 + 260)) for c in range(n)]
    s = sum(sc)
    w = [int(total * x / s) for x in sc]
    for _ in range(5):  # 머리글이 쪼개지지 않게 최소 폭을 맞추고 남는 폭에서 뺀다
        short = sum(max(0, mins[c] - w[c]) for c in range(n))
        if not short:
            break
        rich = [c for c in range(n) if w[c] > mins[c]]
        tot_rich = sum(w[c] - mins[c] for c in rich)
        for c in range(n):
            if w[c] < mins[c]:
                w[c] = mins[c]
        for c in rich:
            w[c] -= int(short * (w[c] - mins[c]) / tot_rich)
    w[-1] += total - sum(w)
    return w

def table(rows, total=9000, vmerge_first=False):
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    is_group = lambda r: n > 2 and r[0].startswith("**") and not any(r[1:])
    w = col_widths([r for r in rows if not is_group(r)], total)
    small = n >= 6
    sz = 16 if small else 18
    x = [f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:tblBorders>{BORDER("BBBBBB")}</w:tblBorders>'
         '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>'
         "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{v}"/>' for v in w) + "</w:tblGrid>"]
    for ri, r in enumerate(rows):
        hdr = ri == 0
        x.append("<w:tr><w:trPr>" + ("<w:tblHeader/>" if hdr else "<w:cantSplit/>") + "</w:trPr>")
        if not hdr and n > 2 and r[0].startswith("**") and not any(r[1:]):  # 묶음 제목 행 (v13): 한 칸으로 합침
            tc = f'<w:tcW w:w="{total}" w:type="dxa"/><w:gridSpan w:val="{n}"/><w:shd w:val="clear" w:color="auto" w:fill="F7F7F7"/>'
            x.append(f"<w:tc><w:tcPr>{tc}{CELLMAR}</w:tcPr>" + para(r[0], sz=sz, jc="left", after=0, line=264) + "</w:tc></w:tr>")
            continue
        for ci, c in enumerate(r):
            tc = f'<w:tcW w:w="{w[ci]}" w:type="dxa"/>'
            if vmerge_first and ci == 0 and not hdr:
                nxt_empty = ri + 1 < len(rows) and rows[ri + 1][0] == ""
                if c == "" and ri > 1:
                    tc += '<w:vMerge/>'
                elif nxt_empty:
                    tc += '<w:vMerge w:val="restart"/>'
            if hdr:
                tc += '<w:shd w:val="clear" w:color="auto" w:fill="EFEFEF"/>'
            tc += CELLMAR + '<w:vAlign w:val="center"/>'
            short = dispw(c) <= 26
            jc = "center" if (hdr or (ci == 0 and n == 2) or short) else "left"
            x.append(f"<w:tc><w:tcPr>{tc}</w:tcPr>" + para(c, sz=sz, jc=jc, after=0, line=264, b=hdr) + "</w:tc>")
        x.append("</w:tr>")
    x.append("</w:tbl>")
    return "".join(x)

# 번호 목록: 목록마다 새 numId (번호 다시 시작)
NUMS = []  # (numId, abstractId)
def new_num(kind, start=1):
    nid = 100 + len(NUMS)
    NUMS.append((nid, 90 if kind == "bul" else 91, start))
    return nid

def lst(items, sz=None):
    xs = []
    ids = {}
    for lvl, kind, text, start in items:
        key = (lvl, kind)
        if key not in ids:
            ids[key] = new_num(kind, start)
        if lvl == 0:
            for k in [k for k in ids if k[0] == 1]:
                del ids[k]
        xs.append(para(text, sz=sz, after=40, num=(ids[key], lvl)))
    return "".join(xs)

def code(buf):
    xs = []
    for k, ln in enumerate(buf):
        after = 0
        rp = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="맑은 고딕"/><w:sz w:val="15"/><w:szCs w:val="15"/>'
        xs.append(f'<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F3F3F3"/><w:spacing w:after="{after}" w:line="240" w:lineRule="auto"/><w:ind w:left="60"/></w:pPr>'
                  f'<w:r><w:rPr>{rp}</w:rPr><w:t xml:space="preserve">{escape(ln) if ln else " "}</w:t></w:r></w:p>')
    xs.append(para("", after=40))
    return "".join(xs)

IMAGES = []  # (rId, filename)
def image(fn, caption, width_dxa, pct=None):
    data = open(f"{FIGS}/{fn}", "rb").read()
    wpx, hpx = struct.unpack(">II", data[16:24])
    cx = int(width_dxa * (pct / 100 if pct else 0.92) * 635)
    cy = int(cx * hpx / wpx)
    rid = f"rIdImg{len(IMAGES) + 1}"
    IMAGES.append((rid, fn))
    k = len(IMAGES)
    drw = (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
           f'<wp:docPr id="{k}" name="Figure {k}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
           '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{k}" name="{fn}"/><pic:cNvPicPr/></pic:nvPicPr>'
           f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
           f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
           "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>")
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="80" w:after="40"/><w:jc w:val="center"/></w:pPr>{drw}</w:p>'
            + para(caption, sz=17, jc="center", after=120, color="555555"))

def render(blocks, width, sz=None):
    xs = []
    for kind, val in blocks:
        if kind == "para":
            xs.append(para(val, sz=sz))
        elif kind == "table":
            xs.append(table(val, total=width, vmerge_first=val[0][0] == "구분" and any(r[0] == "" for r in val)))
            xs.append(para("", after=40, line=200))
        elif kind == "list":
            xs.append(lst(val, sz=sz))
        elif kind == "code":
            xs.append(code(val))
        elif kind == "fig":
            xs.append(image(val[0], val[1], width, val[2] if len(val) > 2 else None))
    return "".join(xs)

def box(inner):
    return (f'<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders>{BORDER("C0C0C0")}</w:tblBorders>'
            '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
            '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="9000"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="9000" w:type="dxa"/>'
            '<w:tcMar><w:top w:w="90" w:type="dxa"/><w:left w:w="130" w:type="dxa"/><w:bottom w:w="90" w:type="dxa"/><w:right w:w="130" w:type="dxa"/></w:tcMar></w:tcPr>'
            f"{inner}<w:p/></w:tc></w:tr></w:tbl>" + para("", after=100))

def add_figs(blocks):
    out = []
    for b in blocks:
        out.append(b)
        if b[0] == "list":
            for key, fig in FIG_AFTER.items():
                if key.startswith("- ") and any(("- " + it[2]).startswith(key) for it in b[1]):
                    out.append(("fig", fig))
        if b[0] == "table":
            for key, fig in FIG_AFTER.items():
                if key.startswith("|") and any(("| " + " | ".join(r) + " |") == key for r in b[1]):
                    out.append(("fig", fig))
    return out

# ---------- 섹션 나누기 ----------
lines = md.split("\n")
secs = []  # (level, title, lines)
cur = None
infence = False
for ln in lines:
    if ln.startswith("```"):
        infence = not infence
    m = None if infence else re.match(r"^(#{1,3}) (.*)", ln)
    if m:
        cur = [len(m.group(1)), m.group(2).strip(), []]
        secs.append(cur)
    elif cur:
        cur[2].append(ln)

body = []
tpl = zipfile.ZipFile(TPL)
doc = tpl.read("word/document.xml").decode("utf-8")
head, rest = doc.split("<w:body>")
segs = re.findall(r"<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>", rest, re.S)
body += segs[:3]  # 제목 3줄
sect = re.search(r"<w:sectPr.*?</w:sectPr>", rest, re.S).group(0)

for lvl, title, sl in secs:
    if lvl == 1:
        continue  # md 제목줄 (양식 제목 사용)
    sid = re.match(r"([\d.]+)", title)
    sid = sid.group(1).rstrip(".") if sid else ""
    is_app = title.startswith("부록")
    body.append(heading(title, 1 if lvl == 2 else 2, pagebreak=title.startswith("부록 A") and not os.environ.get("REPORT_NO_APPENDIX_BREAK")))
    blocks = add_figs(parse(sl))
    if sid == "2":
        k = next(i for i, b in enumerate(blocks) if b[0] == "para" and b[1].startswith("**요약.**"))
        body.append(render(blocks[:k], 9000))
        body.append(para("요약", b=True, after=60, keep=True))
        body.append(box(render(blocks[k + 1:], 8740)))
    elif sid in BOX:
        body.append(box(render(blocks, 8740)))
    else:
        body.append(render(blocks, 9000, sz=18 if is_app else None))

newdoc = head + "<w:body>" + "".join(body) + sect + "</w:body></w:document>"

# ---------- 패키지 ----------
num = tpl.read("word/numbering.xml").decode("utf-8")
def absn(aid, fmt):
    lv = ""
    for l in range(2):
        txt = ("•" if l == 0 else "–") if fmt == "bullet" else f"%{l + 1}."
        nf = "bullet" if fmt == "bullet" else ("decimal" if l == 0 else "lowerLetter")
        left = 360 + 300 * l
        lv += (f'<w:lvl w:ilvl="{l}"><w:start w:val="1"/><w:numFmt w:val="{nf}"/><w:lvlText w:val="{txt}"/><w:lvlJc w:val="left"/>'
               f'<w:pPr><w:ind w:left="{left}" w:hanging="260"/></w:pPr></w:lvl>')
    return f'<w:abstractNum w:abstractNumId="{aid}"><w:multiLevelType w:val="hybridMultilevel"/>{lv}</w:abstractNum>'
abs_x = absn(90, "bullet") + absn(91, "decimal")
num = re.sub(r"(<w:abstractNum )", abs_x + r"\1", num, count=1)
nums_x = "".join(f'<w:num w:numId="{nid}"><w:abstractNumId w:val="{aid}"/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="{st}"/></w:lvlOverride></w:num>' for nid, aid, st in NUMS)
num = num.replace("</w:numbering>", nums_x + "</w:numbering>")

rels = tpl.read("word/_rels/document.xml.rels").decode("utf-8")
rels = rels.replace("</Relationships>", "".join(
    f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{fn}"/>' for rid, fn in IMAGES) + "</Relationships>")
ct = tpl.read("[Content_Types].xml").decode("utf-8")
if 'Extension="png"' not in ct:
    ct = ct.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"', 1)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for it in tpl.infolist():
        n = it.filename
        if n == "word/document.xml":
            z.writestr(n, newdoc)
        elif n == "word/numbering.xml":
            z.writestr(n, num)
        elif n == "word/_rels/document.xml.rels":
            z.writestr(n, rels)
        elif n == "[Content_Types].xml":
            z.writestr(n, ct)
        else:
            z.writestr(it, tpl.read(n))
    for rid, fn in IMAGES:
        z.write(f"{FIGS}/{fn}", f"word/media/{fn}")
print("ok", OUT, "figs", len(IMAGES), "lists", len(NUMS))
left = re.findall(r"\[확인 필요[^\]]*\]", md)
print("확인 필요", len(left)); [print(" ", s) for s in left]
