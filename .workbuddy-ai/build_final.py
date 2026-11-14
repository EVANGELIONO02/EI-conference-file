# -*- coding: utf-8 -*-
"""Rebuild slide1.xml : 3 horizontal bands (top-down), unified grid, re-aligned.

All wording / figures are kept verbatim from the user's own edited file:
  stage1  4 step chips + 2 branch chips + wind.png + wind_k-means.png (+ 2 captions)
  stage2  2 step chips + Diffusion principle figure
  stage3  3 step chips + 3 parallel comparison chips

Layout contract
---------------
band          X=20  W=920
stage title   X=34  Y=band+7   14 pt bold, stage colour
hairline      X=32  Y=band+37
step chip     X=44  W=140 H=28   stage-colour fill, white 9.5 pt text   (col A)
content col   X=230 / X=690      figures & branch chips
sub chip      W=140 H=28         white fill, stage-colour border/text 8.5 pt

Arrow grammar
-------------
vertical   = main flow (step chain, stage chain)      -> down
horizontal = content (chip -> figure, chip -> branch)  -> right
Every horizontal run sits in its own empty channel, so no line crosses
another line, a text box or a picture.
"""
import zipfile, re, os

SRC = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图_修改版.pptx'
TMP = os.path.join(os.environ.get('TEMP', '/tmp'), 'hb_final.pptx')

W, H = 960, 1100
EMU = 12700

C1, C2, C3 = '2F8F7A', 'C08A3E', '3E77A8'
GRAY = '8C9BA5'
CHIP_TX = 'FFFFFF'
IMG_LN, CAP_TX = 'C7DED8', '5C7A72'
FONT = '微软雅黑'

NODE_W, NODE_H = 140, 28
CHIP_X = 44            # col A : step chips
SUB_X = 230            # col B : branch chips
FIG_X = 690            # col C : figures
SPINE_X = CHIP_X + NODE_W / 2      # 114
GAP = 6
CHIP_ADJ = 20000

_id = [200]


def nid():
    _id[0] += 1
    return _id[0]


def e(v):
    return int(round(v * EMU))


def xfrm(x, y, w, h):
    return f'<a:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(w)}" cy="{e(h)}"/></a:xfrm>'


def run(text, sz, color, bold=True):
    b = ' b="1"' if bold else ''
    return (f'<a:r><a:rPr lang="zh-CN" sz="{sz}"{b}><a:solidFill><a:srgbClr val="{color}"/>'
            f'</a:solidFill><a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/></a:rPr>'
            f'<a:t>{text}</a:t></a:r>')


def _body(fill, line, lw, dash=False):
    s = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
    if line:
        s += (f'<a:ln w="{e(lw)}"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill>'
              f'<a:prstDash val="{"dash" if dash else "solid"}"/></a:ln>')
    else:
        s += '<a:ln><a:noFill/></a:ln>'
    return s


def sp_rect(x, y, w, h, fill=None, line=None, lw=0.75, text=None,
            sz=None, color=None, algn='ctr', anchor='ctr', bold=True, dash=False,
            wrap='square'):
    tx = ''
    if text is not None:
        tx = (f'<p:txBody><a:bodyPr wrap="{wrap}" anchor="{anchor}" anchorCtr="0">'
              f'<a:normAutofit/></a:bodyPr><a:p><a:pPr algn="{algn}"/>'
              f'{run(text, sz, color, bold)}</a:p></p:txBody>')
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name=""/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'{_body(fill, line, lw, dash)}</p:spPr>{tx}</p:sp>')


def sp_round(x, y, w, h, fill=None, line=None, lw=0.75, adj=CHIP_ADJ, text=None,
             sz=None, color=None, algn='ctr', anchor='ctr', bold=True, wrap='none'):
    tx = ''
    if text is not None:
        tx = (f'<p:txBody><a:bodyPr wrap="{wrap}" anchor="{anchor}" anchorCtr="0">'
              f'<a:normAutofit/></a:bodyPr><a:p><a:pPr algn="{algn}"/>'
              f'{run(text, sz, color, bold)}</a:p></p:txBody>')
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name=""/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="roundRect"><a:avLst>'
            f'<a:gd name="adj" fmla="val {adj}"/></a:avLst></a:prstGeom>'
            f'{_body(fill, line, lw, False)}</p:spPr>{tx}</p:sp>')


def sp_line(x, y, w, h, color, lw=1.5, arrow=True):
    tail = '<a:tailEnd type="triangle"/>' if arrow else ''
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name=""/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:noFill/><a:ln w="{e(lw)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:prstDash val="solid"/>{tail}</a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr/><a:p><a:pPr/><a:endParaRPr/></a:p></p:txBody></p:sp>')


def sp_pic(x, y, w, h, rid):
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="{nid()}" name=""/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch/></p:blipFill>'
            f'<p:spPr>{xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')


shapes = []

# ---------------- bands ----------------
BX, BW = 20, 920
B1_Y, B1_H = 20, 432
B2_Y, B2_H = 476, 342
B3_Y, B3_H = 842, 240

for y, h, color in [(B1_Y, B1_H, C1), (B2_Y, B2_H, C2), (B3_Y, B3_H, C3)]:
    shapes.append(sp_round(BX, y, BW, h, fill='FFFFFF', line=color, lw=2, adj=3000))

for (y, color, label) in [(B1_Y, C1, '阶段1：数据清洗'),
                          (B2_Y, C2, '阶段2：数据填充'),
                          (B3_Y, C3, '阶段3：功率预测')]:
    shapes.append(sp_rect(BX + 14, y + 7, 300, 22, text=label, sz=1400, color=color,
                           algn='l', anchor='ctr'))
    shapes.append(sp_line(BX + 12, y + 37, BW - 24, 0, 'DCE4E6', lw=0.75, arrow=False))

# ================= stage 1 =================
# col A (x=44) : 4 step chips      col B (x=230) : the two scatter figures
# col C (x=690) : the two branch chips of step 4
S1_ROWS = [62, 186, 236, 380]          # chip tops -> centres 76 / 200 / 250 / 394
S1_TITLES = ['第一步 数据预处理', '第二步 归一化',
             '第三步 按风速大小划分数据', '第四步 构造缺失数据']
for title, y in zip(S1_TITLES, S1_ROWS):
    shapes.append(sp_round(CHIP_X, y, NODE_W, NODE_H, fill=C1, line='FFFFFF', lw=1.0,
                           text=title, sz=950, color=CHIP_TX))

for (a, b) in [(0, 1), (1, 2), (2, 3)]:
    ya, yb = S1_ROWS[a], S1_ROWS[b]
    shapes.append(sp_line(SPINE_X, ya + NODE_H + 2, 0, yb - ya - NODE_H - 4, C1, lw=1.5))

# ---- wind.png (step1) ----
FIG_W = 190
WIND_H = FIG_W * 1009 / 1654.0     # 115.9
KM_H = FIG_W * 1024 / 1824.0       # 106.7
F1_Y, F2_Y = 60, 234

shapes.append(sp_line(CHIP_X + NODE_W + GAP, S1_ROWS[0] + NODE_H / 2,
                      SUB_X - CHIP_X - NODE_W - GAP, 0, C1, lw=1.5))
shapes.append(sp_pic(SUB_X, F1_Y, FIG_W, WIND_H, 'rId2'))
shapes.append(sp_rect(SUB_X, F1_Y, FIG_W, WIND_H, fill=None, line=IMG_LN, lw=0.75))
shapes.append(sp_rect(SUB_X, F1_Y + WIND_H + 3, FIG_W, 13, text='Wind',
                       sz=700, color=CAP_TX, bold=False))

# ---- wind k-means (step3) ----
shapes.append(sp_line(CHIP_X + NODE_W + GAP, S1_ROWS[2] + NODE_H / 2,
                      SUB_X - CHIP_X - NODE_W - GAP, 0, C1, lw=1.5))
shapes.append(sp_pic(SUB_X, F2_Y, FIG_W, KM_H, 'rId3'))
shapes.append(sp_rect(SUB_X, F2_Y, FIG_W, KM_H, fill=None, line=IMG_LN, lw=0.75))
shapes.append(sp_rect(SUB_X, F2_Y + KM_H + 3, FIG_W, 13, text='Classified Wind',
                       sz=700, color=CAP_TX, bold=False))

# ---- step4 -> two branch chips (right-angle bus, col C) ----
S1_BUS_X = 660
S1_SUB_X = 690
S1_SUB_Y = [366, 406]              # centres 380 / 420
shapes.append(sp_line(CHIP_X + NODE_W + GAP, S1_ROWS[3] + NODE_H / 2,
                      S1_BUS_X - CHIP_X - NODE_W - GAP, 0, C1, lw=1.5, arrow=False))
shapes.append(sp_line(S1_BUS_X, S1_SUB_Y[0] + NODE_H / 2, 0,
                      S1_SUB_Y[1] - S1_SUB_Y[0], C1, lw=1.5, arrow=False))
for y, label in zip(S1_SUB_Y, ['构造随机点缺失', '构造随机连续缺失']):
    shapes.append(sp_line(S1_BUS_X, y + NODE_H / 2, S1_SUB_X - S1_BUS_X, 0, C1, lw=1.5))
    shapes.append(sp_round(S1_SUB_X, y, NODE_W, NODE_H, fill='FFFFFF', line=C1, lw=1.25,
                           text=label, sz=850, color=C1))

# ================= stage 2 =================
S2_R1, S2_R2 = 548, 596            # centres 562 / 610
S2_FIG_X, S2_FIG_W = 230, 560
S2_FIG_H = S2_FIG_W * 410 / 809.0  # 283.8
S2_FIG_Y = 520

shapes.append(sp_round(CHIP_X, S2_R1, NODE_W, NODE_H, fill=C2, line='FFFFFF', lw=1.0,
                       text='第一步 diffusion模型训练', sz=950, color=CHIP_TX))
shapes.append(sp_round(CHIP_X, S2_R2, NODE_W, NODE_H, fill=C2, line='FFFFFF', lw=1.0,
                       text='第二步 扩充缺失数据', sz=950, color=CHIP_TX))
shapes.append(sp_line(SPINE_X, S2_R1 + NODE_H + 2, 0, S2_R2 - S2_R1 - NODE_H - 4, C2, lw=1.5))
shapes.append(sp_line(CHIP_X + NODE_W + GAP, S2_R1 + NODE_H / 2,
                      S2_FIG_X - CHIP_X - NODE_W - GAP, 0, C2, lw=1.5))
shapes.append(sp_pic(S2_FIG_X, S2_FIG_Y, S2_FIG_W, S2_FIG_H, 'rId4'))
shapes.append(sp_rect(S2_FIG_X, S2_FIG_Y, S2_FIG_W, S2_FIG_H, fill=None, line=IMG_LN, lw=0.75))

# ================= stage 3 =================
S3_R1, S3_R2, S3_R3 = 886, 934, 982        # centres 900 / 948 / 996
S3_BUS_Y = 1026
S3_SUB_X = [230, 460, 690]
S3_SUB_Y = 1040

shapes.append(sp_round(CHIP_X, S3_R1, NODE_W, NODE_H, fill=C3, line='FFFFFF', lw=1.0,
                       text='第一步 训练 LSTM 模型', sz=950, color=CHIP_TX))
shapes.append(sp_round(CHIP_X, S3_R2, NODE_W, NODE_H, fill=C3, line='FFFFFF', lw=1.0,
                       text='第二步 扩充数据进行功率预测', sz=950, color=CHIP_TX))
shapes.append(sp_round(CHIP_X, S3_R3, NODE_W, NODE_H, fill=C3, line='FFFFFF', lw=1.0,
                       text='第三步 对比不同填充方法', sz=950, color=CHIP_TX))
for a, b in [(S3_R1, S3_R2), (S3_R2, S3_R3)]:
    shapes.append(sp_line(SPINE_X, a + NODE_H + 2, 0, b - a - NODE_H - 4, C3, lw=1.5))

# step3 -> three parallel comparison chips (drop bus below)
shapes.append(sp_line(SPINE_X, S3_R3 + NODE_H + 2, 0, S3_BUS_Y - S3_R3 - NODE_H - 4,
                       C3, lw=1.5, arrow=False))
shapes.append(sp_line(S3_SUB_X[0] + NODE_W / 2, S3_BUS_Y,
                      S3_SUB_X[2] + NODE_W / 2 - (S3_SUB_X[0] + NODE_W / 2),
                      0, C3, lw=1.5, arrow=False))
for x, label in zip(S3_SUB_X, ['线性插值',
                                '无气象条件的 Diffusion 模型',
                                '有气象条件的 Diffusion 模型']):
    shapes.append(sp_line(x + NODE_W / 2, S3_BUS_Y, 0, S3_SUB_Y - S3_BUS_Y, C3, lw=1.5))
    shapes.append(sp_round(x, S3_SUB_Y, NODE_W, NODE_H, fill='FFFFFF', line=C3, lw=1.25,
                           text=label, sz=850, color=C3))

# ---------------- inter-band flow arrows ----------------
shapes.append(sp_line(BX + BW / 2, B1_Y + B1_H + 6, 0, B2_Y - B1_Y - B1_H - 12, GRAY, lw=1.75))
shapes.append(sp_line(BX + BW / 2, B2_Y + B2_H + 6, 0, B3_Y - B2_Y - B2_H - 12, GRAY, lw=1.75))

slide_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
    '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/>'
    '</p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
    '</p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
    '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    + ''.join(shapes) +
    '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')

# ---------------- repackage ----------------
z = zipfile.ZipFile(SRC, 'r')
items = z.infolist()
data = {i.filename: z.read(i.filename) for i in items}
z.close()

data['ppt/slides/slide1.xml'] = slide_xml.encode('utf-8')
pres = data['ppt/presentation.xml'].decode('utf-8')
pres = re.sub(r'<p:sldSz[^/]*/>', f'<p:sldSz cx="{e(W)}" cy="{e(H)}"/>', pres)
data['ppt/presentation.xml'] = pres.encode('utf-8')

if os.path.exists(TMP):
    os.remove(TMP)
zo = zipfile.ZipFile(TMP, 'w', zipfile.ZIP_DEFLATED)
for i in items:
    zo.writestr(i, data[i.filename])
zo.close()
print('built:', TMP, os.path.getsize(TMP), 'shapes:', len(shapes))
