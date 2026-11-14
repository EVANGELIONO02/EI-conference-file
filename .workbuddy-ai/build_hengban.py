# -*- coding: utf-8 -*-
"""Rebuild slide1.xml : 3 horizontal bands (top-down), unified node grid.

Grid (pt)
---------
col A  44..184   主流程芯片 (NODE_W=140, NODE_H=28)
col B  230..490  内容位：阶段1 = 虚线预留框(W=260)；阶段2 = 文案框(W=480)
col C  400..660  分支说明位 (W=260)
figs   520..912  插图组（第三步），与第三步同行居中
rows   64 / 104 / 144 (行距 40)，模块 256，分支 236 / 276

Arrow grammar
-------------
主干（步骤链）= 竖向，向下；
内容（芯片->说明位 / 说明位->插图 / 模块->分支）= 横向，向右。
所有节点框统一 140x28。
"""
import zipfile, re, os

SRC = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图_修改版.pptx'
TMP = os.path.join(os.environ.get('TEMP', '/tmp'), 'hb_build.pptx')

W, H = 960, 828
EMU = 12700

C1, C2, C3 = '2F8F7A', 'C08A3E', '3E77A8'
GRAY = '8C9BA5'
CHIP_TX = 'FFFFFF'
COPY_BG, COPY_LN, COPY_TX = 'EEF6F4', 'C7DED8', '33514A'
SLOT_LN = 'C7DED8'
IMG_LN, CAP_TX = 'C7DED8', '5C7A72'
FONT = '微软雅黑'

# ---- 统一节点尺寸 ----
NODE_W, NODE_H = 140, 28
CHIP_X = 44          # col A
CONT_X = 230         # col B
SLOT_W = 240         # 说明位统一宽 (230..470)
BR_SLOT_X, BR_SLOT_W = 400, 240   # 分支说明位 (400..640)
SPINE_X = CHIP_X + NODE_W / 2     # 114
GAP = 6
CHIP_ADJ = 20000

_id = [100]


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


def _body(fill, line, lw, dash):
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
B1_Y, B1_H = 20, 284
B2_Y, B2_H = 328, 156
B3_Y, B3_H = 508, 300

shapes.append(sp_round(BX, B1_Y, BW, B1_H, fill='FFFFFF', line=C1, lw=2, adj=3000))
shapes.append(sp_round(BX, B2_Y, BW, B2_H, fill='FFFFFF', line=C2, lw=2, adj=3000))
shapes.append(sp_round(BX, B3_Y, BW, B3_H, fill='FFFFFF', line=C3, lw=2, adj=3000))

for (y, color, label) in [(B1_Y, C1, '阶段1：数据清洗'),
                          (B2_Y, C2, '阶段2：数据填充'),
                          (B3_Y, C3, '阶段3：功率预测')]:
    shapes.append(sp_rect(BX + 14, y + 7, 300, 22, text=label, sz=1400, color=color,
                           algn='l', anchor='ctr'))
    shapes.append(sp_line(BX + 12, y + 37, BW - 24, 0, 'DCE4E6', lw=0.75, arrow=False))

# ================= stage 1 =================
R1, R2, R3 = 64, 104, 144

for y, title in [(R1, '第一步 数据预处理'),
                 (R2, '第二步 归一化'),
                 (R3, '第三步 按风速大小划分数据')]:
    shapes.append(sp_round(CHIP_X, y, NODE_W, NODE_H, fill=C1, line='FFFFFF', lw=1.0,
                           text=title, sz=950, color=CHIP_TX))
    # 芯片 -> 说明位（横向向右，落在框上，不悬空）
    shapes.append(sp_line(CHIP_X + NODE_W + GAP, y + NODE_H / 2,
                          CONT_X - CHIP_X - NODE_W - GAP, 0, C1, lw=1.5))
    shapes.append(sp_rect(CONT_X, y, SLOT_W, NODE_H, fill=None, line=SLOT_LN, lw=0.75, dash=True))

# 主干竖向连接：步骤1 -> 步骤2 -> 步骤3
shapes.append(sp_line(SPINE_X, R1 + NODE_H + 2, 0, R2 - R1 - NODE_H - 4, C1, lw=1.5))
shapes.append(sp_line(SPINE_X, R2 + NODE_H + 2, 0, R3 - R2 - NODE_H - 4, C1, lw=1.5))

# 插图（第三步：按风速大小划分数据）
IW = 170
h1 = IW * 85.4 / 140.0
h2 = IW * 78.6 / 140.0
IX1, IX2 = 560, 746
FIG_BLOCK_H = h1 + 4 + 13
FIG_Y = R3 + NODE_H / 2 - FIG_BLOCK_H / 2      # 与第三步同行、垂直居中
shapes.append(sp_line(CONT_X + SLOT_W + GAP, R3 + NODE_H / 2,
                      IX1 - CONT_X - SLOT_W - GAP, 0, C1, lw=1.5))
shapes.append(sp_pic(IX1, FIG_Y, IW, h1, 'rId1'))
shapes.append(sp_rect(IX1, FIG_Y, IW, h1, fill=None, line=IMG_LN, lw=0.75))
shapes.append(sp_pic(IX2, FIG_Y + (h1 - h2) / 2, IW, h2, 'rId2'))
shapes.append(sp_rect(IX2, FIG_Y + (h1 - h2) / 2, IW, h2, fill=None, line=IMG_LN, lw=0.75))
CAP_Y = FIG_Y + h1 + 4
shapes.append(sp_rect(IX1, CAP_Y, IW, 13, text='Wind', sz=700, color=CAP_TX, bold=False))
shapes.append(sp_rect(IX2, CAP_Y, IW, 13, text='Wind K-means', sz=700, color=CAP_TX, bold=False))

# 构造缺失数据（主干延续）+ 两个并列分支
BR_A_Y, BR_B_Y = 214, 254
MOD_Y = 241
BUS_X = 206

shapes.append(sp_line(SPINE_X, R3 + NODE_H + 2, 0, MOD_Y - R3 - NODE_H - 4, C1, lw=1.5))
shapes.append(sp_round(CHIP_X, MOD_Y, NODE_W, NODE_H, fill=C1, line='FFFFFF', lw=1.0,
                       text='构造缺失数据', sz=950, color=CHIP_TX))

# 分支母线：模块右侧水平段 -> 竖干 -> 两条水平箭头
shapes.append(sp_line(CHIP_X + NODE_W + GAP, MOD_Y + NODE_H / 2,
                      BUS_X - CHIP_X - NODE_W - GAP, 0, C1, lw=1.5, arrow=False))
shapes.append(sp_line(BUS_X, BR_A_Y + NODE_H / 2, 0, BR_B_Y - BR_A_Y, C1, lw=1.5, arrow=False))
for y, label in [(BR_A_Y, '构造随机点缺失'), (BR_B_Y, '构造随机连续缺失')]:
    shapes.append(sp_line(BUS_X, y + NODE_H / 2, CONT_X - BUS_X, 0, C1, lw=1.5))
    shapes.append(sp_round(CONT_X, y, NODE_W, NODE_H, fill='FFFFFF', line=C1, lw=1.25,
                           text=label, sz=900, color=C1))
    shapes.append(sp_line(CONT_X + NODE_W + GAP, y + NODE_H / 2,
                          BR_SLOT_X - CONT_X - NODE_W - GAP, 0, C1, lw=1.5))
    shapes.append(sp_rect(BR_SLOT_X, y, BR_SLOT_W, NODE_H, fill=None, line=SLOT_LN,
                          lw=0.75, dash=True))

# ================= stage 2 =================
S2_R1, S2_R2 = 384, 438
S2_TXT_X, S2_TXT_W, S2_TXT_H = CONT_X, 480, 44

for y, title, copy in [
    (S2_R1, '第一步 diffusion模型训练',
     '训练条件扩散模型。模型输入包括部分观测的历史功率序列、缺失掩码以及气象工况标签。'
     '模型学习在观测条件约束下重构缺失值并预测未来功率的联合分布。'),
    (S2_R2, '第二步 扩充缺失数据',
     '在每个气象工况内，分别生成包含随机点缺失和随机连续缺失的样本集，扩充训练数据。'
     '缺失样本与完整样本共同参与模型训练，增强模型对缺失场景的适应能力。'),
]:
    shapes.append(sp_round(CHIP_X, y, NODE_W, NODE_H, fill=C2, line='FFFFFF', lw=1.0,
                           text=title, sz=900, color=CHIP_TX))
    shapes.append(sp_line(CHIP_X + NODE_W + GAP, y + NODE_H / 2,
                          S2_TXT_X - CHIP_X - NODE_W - GAP, 0, C2, lw=1.5))
    shapes.append(sp_rect(S2_TXT_X, y - (S2_TXT_H - NODE_H) / 2, S2_TXT_W, S2_TXT_H,
                          fill=COPY_BG, line=COPY_LN, lw=0.75, text=copy, sz=900,
                          color=COPY_TX, algn='l', anchor='ctr'))

shapes.append(sp_line(SPINE_X, S2_R1 + NODE_H + 2, 0, S2_R2 - S2_R1 - NODE_H - 4, C2, lw=1.5))

# ================= stage 3 =================
S3_R1, S3_R2, S3_R3 = 558, 598, 638
S3_SLOT_X, S3_SLOT_W = CONT_X, 480     # 230..710，与阶段2文案框同宽
S3_BUS_X = 206                        # 分支母线竖干（与阶段1一致）
S3_BR_X, S3_BR_Y = 230, [682, 722, 762]   # 三个对比框
S3_BR_SLOT_X, S3_BR_SLOT_W = 400, 480     # 400..880

# ① ②：芯片 + 右侧说明位
for y, title in [(S3_R1, '第一步 训练 LSTM 模型'),
                 (S3_R2, '第二步 扩充数据进行功率预测')]:
    shapes.append(sp_round(CHIP_X, y, NODE_W, NODE_H, fill=C3, line='FFFFFF', lw=1.0,
                           text=title, sz=900, color=CHIP_TX))
    shapes.append(sp_line(CHIP_X + NODE_W + GAP, y + NODE_H / 2,
                          S3_SLOT_X - CHIP_X - NODE_W - GAP, 0, C3, lw=1.5))
    shapes.append(sp_rect(S3_SLOT_X, y, S3_SLOT_W, NODE_H, fill=None, line=SLOT_LN,
                          lw=0.75, dash=True))

# ③ 对比节点（下游为三分支，故不配说明位，与阶段1「构造缺失数据」层级一致）
shapes.append(sp_round(CHIP_X, S3_R3, NODE_W, NODE_H, fill=C3, line='FFFFFF', lw=1.0,
                       text='第三步 对比不同填充方法', sz=900, color=CHIP_TX))

# 主干竖向连接：① -> ② -> ③
shapes.append(sp_line(SPINE_X, S3_R1 + NODE_H + 2, 0, S3_R2 - S3_R1 - NODE_H - 4, C3, lw=1.5))
shapes.append(sp_line(SPINE_X, S3_R2 + NODE_H + 2, 0, S3_R3 - S3_R2 - NODE_H - 4, C3, lw=1.5))

# 第三步 -> 分支母线（横向段无箭头，方向由三条分支箭头承载）
shapes.append(sp_line(CHIP_X + NODE_W + GAP, S3_R3 + NODE_H / 2,
                      S3_BUS_X - CHIP_X - NODE_W - GAP, 0, C3, lw=1.5, arrow=False))
shapes.append(sp_line(S3_BUS_X, S3_R3 + NODE_H / 2, 0,
                      S3_BR_Y[2] + NODE_H / 2 - (S3_R3 + NODE_H / 2), C3, lw=1.5, arrow=False))

for y, label in zip(S3_BR_Y, ['线性插值',
                              '无气象条件的 Diffusion 模型',
                              '有气象条件的 Diffusion 模型']):
    shapes.append(sp_line(S3_BUS_X, y + NODE_H / 2, S3_BR_X - S3_BUS_X, 0, C3, lw=1.5))
    shapes.append(sp_round(S3_BR_X, y, NODE_W, NODE_H, fill='FFFFFF', line=C3, lw=1.25,
                           text=label, sz=850, color=C3))
    shapes.append(sp_line(S3_BR_X + NODE_W + GAP, y + NODE_H / 2,
                          S3_BR_SLOT_X - S3_BR_X - NODE_W - GAP, 0, C3, lw=1.5))
    shapes.append(sp_rect(S3_BR_SLOT_X, y, S3_BR_SLOT_W, NODE_H, fill=None, line=SLOT_LN,
                          lw=0.75, dash=True))

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
