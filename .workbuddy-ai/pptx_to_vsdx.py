# -*- coding: utf-8 -*-
"""Convert the flow-chart PPTX slide into a native, fully editable Visio .vsdx.

Why not Visio COM?
------------------
Visio's Page.Drop() takes its picture argument as a *byref* VARIANT, which
pywin32 / PowerShell / VBScript cannot marshal here.  Writing the VSDX package
directly is both more reliable and gives full control.

Result: every band, chip, arrow and picture becomes an independent Visio
shape (Type='Shape' / Type='Foreign'), so the user can click any box and
re-type, recolour or move it.

Coordinate mapping
------------------
PPT : EMU, origin top-left,  y grows downwards
Visio: inches, origin bottom-left, y grows upwards
      visio_x = ppt_x / 72
      visio_y = (PAGE_H_pt - ppt_y) / 72
"""
import zipfile, re, os, shutil

SRC = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图_修改版.pptx'
OUT = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图.vsdx'
EMU = 12700.0
NS = 'http://schemas.microsoft.com/office/visio/2012/main'
RNS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
VRNS = 'http://schemas.microsoft.com/visio/2010/relationships'

# ------------------------------------------------------------------ read PPTX
z = zipfile.ZipFile(SRC)
slide = z.read('ppt/slides/slide1.xml').decode('utf-8')
rels_x = z.read('ppt/slides/_rels/slide1.xml.rels').decode('utf-8')
pres = z.read('ppt/presentation.xml').decode('utf-8')

rel_map = dict(re.findall(r'Id="(\w+)"[^>]*Target="([^"]+)"', rels_x))
m = re.search(r'<p:sldSz[^/]*cx="(\d+)"[^/]*cy="(\d+)"', pres)
PAGE_W_PT = int(m.group(1)) / EMU
PAGE_H_PT = int(m.group(2)) / EMU
PAGE_W_IN = PAGE_W_PT / 72.0
PAGE_H_IN = PAGE_H_PT / 72.0

body = slide[slide.find('<p:spTree>'):]
items = re.findall(r'<p:(?:sp|pic)>(?:(?!</p:(?:sp|pic)>).)*</p:(?:sp|pic)>', body, re.S)

# media payloads (name -> bytes) taken straight from the PPTX
media = {}
for name in z.namelist():
    if name.startswith('ppt/media/'):
        media[os.path.basename(name)] = z.read(name)
z.close()


def xfrm_of(s):
    m = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', s)
    return tuple(int(v) / EMU for v in m.groups()) if m else None


def _spPr(s):
    mm = re.search(r'<p:spPr>.*?</p:spPr>', s, re.S)
    return mm.group(0) if mm else ''


def fill_color(s):
    t = _spPr(s)
    if '<a:noFill/>' in t.split('<a:ln')[0]:
        return None
    cm = re.search(r'<a:solidFill><a:srgbClr val="(\w{6})"', t)
    return cm.group(1) if cm else None


def line_color(s):
    t = _spPr(s)
    if '<a:ln><a:noFill/></a:ln>' in t:
        return None
    cm = re.search(r'<a:ln[^>]*>(?:(?!</a:ln>).)*?<a:srgbClr val="(\w{6})"', t, re.S)
    return cm.group(1) if cm else None


def line_width_in(s):
    t = _spPr(s)
    mw = re.search(r'<a:ln w="(\d+)"', t)
    pt = (int(mw.group(1)) / EMU) if mw else 0.75
    return pt / 72.0


def text_of(s):
    return ''.join(re.findall(r'<a:t>([^<]*)</a:t>', s))


def run_style(s):
    t = re.search(r'<a:rPr[^>]*sz="(\d+)"([^>]*)>(.*?)</a:rPr>', s, re.S)
    if not t:
        return None
    size_in = (int(t.group(1)) / 100.0) / 72.0        # pt -> inch
    bold = ' b="1"' in t.group(2)
    cm = re.search(r'<a:srgbClr val="(\w{6})"', t.group(3))
    colour = cm.group(1) if cm else '000000'
    fm = re.search(r'typeface="([^"]+)"', t.group(3))
    font = fm.group(1) if fm else '微软雅黑'
    return size_in, bold, colour, font


def align_of(s):
    am = re.search(r'<a:pPr[^>]*algn="(\w+)"', s)
    return {'l': 0, 'ctr': 1, 'r': 2}[am.group(1) if am else 'l']


def anchor_of(s):
    am = re.search(r'<a:bodyPr[^>]*anchor="(\w+)"', s)
    return {'t': 0, 'ctr': 1, 'b': 2}[am.group(1) if am else 't']


# ------------------------------------------------------------------ helpers
def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;'))


def hexcol(h):
    return '#' + h.upper()


def num(v):
    return ('%.10f' % v).rstrip('0').rstrip('.')


def vx(pt):
    return pt / 72.0


def vy(pt):
    return (PAGE_H_PT - pt) / 72.0


def geom_rect(w_in, h_in):
    """Rectangle geometry with absolute inch coordinates (Visio convention)."""
    return ("<Section N='Geometry' IX='0'>"
            "<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/>"
            "<Cell N='NoShow' V='0'/><Cell N='NoSnap' V='0'/>"
            "<Cell N='NoQuickDrag' V='0'/>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            "<Row T='LineTo' IX='2'><Cell N='X' V='%s'/><Cell N='Y' V='0'/></Row>"
            "<Row T='LineTo' IX='3'><Cell N='X' V='%s'/><Cell N='Y' V='%s'/></Row>"
            "<Row T='LineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='%s'/></Row>"
            "<Row T='LineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            "</Section>" % (num(w_in), num(w_in), num(h_in), num(h_in)))


def geom_line(x1, y1, x2, y2):
    """Two-point polyline in absolute local inches (no rotation involved).

    The shape's Pin is the mid-point of the segment and the local origin is
    the bounding-box corner, so the two points are simply offset from it.
    """
    return ("<Section N='Geometry' IX='0'>"
            "<Cell N='NoFill' V='1'/><Cell N='NoLine' V='0'/>"
            "<Cell N='NoShow' V='0'/><Cell N='NoSnap' V='0'/>"
            "<Cell N='NoQuickDrag' V='0'/>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='%s'/><Cell N='Y' V='%s'/></Row>"
            "<Row T='LineTo' IX='2'><Cell N='X' V='%s'/><Cell N='Y' V='%s'/></Row>"
            "</Section>" % (num(x1), num(y1), num(x2), num(y2)))



def shape_xml(sid, name, x, y, w, h, fill, lncol, lnw, text, style, algn, anchor,
              radius_in=None, geom=None, extra=''):
    """Build one Visio <Shape>."""
    cx, cy = vx(x + w / 2.0), vy(y + h / 2.0)
    W, H = vx(w), (h / 72.0)

    cells = []
    cells.append('<Cell N=\'PinX\' V=\'%s\'/>' % num(cx))
    cells.append('<Cell N=\'PinY\' V=\'%s\'/>' % num(cy))
    cells.append('<Cell N=\'Width\' V=\'%s\'/>' % num(W))
    cells.append('<Cell N=\'Height\' V=\'%s\'/>' % num(H))
    cells.append('<Cell N=\'LocPinX\' V=\'%s\' F=\'Width*0.5\'/>' % num(W / 2.0))
    cells.append('<Cell N=\'LocPinY\' V=\'%s\' F=\'Height*0.5\'/>' % num(H / 2.0))
    cells.append('<Cell N=\'Angle\' V=\'0\' U=\'DEG\'/>')
    cells.append('<Cell N=\'FlipX\' V=\'0\'/><Cell N=\'FlipY\' V=\'0\'/>')
    cells.append('<Cell N=\'ResizeMode\' V=\'0\'/>')
    # line
    cells.append('<Cell N=\'LineWeight\' V=\'%s\'/>' % num(lnw))
    cells.append('<Cell N=\'LineColor\' V=\'%s\'/>' % hexcol(lncol or '000000'))
    cells.append('<Cell N=\'LinePattern\' V=\'%s\'/>' % ('1' if lncol else '0'))
    # fill
    cells.append('<Cell N=\'FillForegnd\' V=\'%s\'/>' % hexcol(fill or 'FFFFFF'))
    cells.append('<Cell N=\'FillBkgnd\' V=\'%s\'/>' % hexcol(fill or 'FFFFFF'))
    cells.append('<Cell N=\'FillPattern\' V=\'%s\'/>' % ('1' if fill else '0'))
    cells.append('<Cell N=\'FillGradientEnabled\' V=\'0\'/>')
    if radius_in:
        cells.append('<Cell N=\'Rounding\' V=\'%s\' U=\'IN\'/>' % num(radius_in))
    # text block
    cells.append('<Cell N=\'TxtPinX\' V=\'%s\' F=\'Inh\'/>' % num(W / 2.0))
    cells.append('<Cell N=\'TxtPinY\' V=\'%s\' F=\'Inh\'/>' % num(H / 2.0))
    cells.append('<Cell N=\'TxtAngle\' V=\'0\'/>')
    cells.append('<Cell N=\'VerticalAlign\' V=\'%s\'/>' % anchor)
    cells.append('<Cell N=\'TextBkgnd\' V=\'0\'/>')
    cells.append('<Cell N=\'TextDirection\' V=\'0\'/>')
    cells.append('<Cell N=\'TxtWidth\' V=\'%s\' F=\'Inh\'/>' % num(W))
    cells.append('<Cell N=\'TxtHeight\' V=\'%s\' F=\'Inh\'/>' % num(H))
    cells.append('<Cell N=\'TxtLocPinX\' V=\'%s\' F=\'Inh\'/>' % num(W / 2.0))
    cells.append('<Cell N=\'TxtLocPinY\' V=\'%s\' F=\'Inh\'/>' % num(H / 2.0))
    cells.append('<Cell N=\'VerticalAlign\' V=\'%s\'/>' % anchor)
    cells.append('<Cell N=\'LeftMargin\' V=\'0\'/><Cell N=\'RightMargin\' V=\'0\'/>')
    cells.append('<Cell N=\'TopMargin\' V=\'0\'/><Cell N=\'BottomMargin\' V=\'0\'/>')

    body = ''.join(cells)

    if style:
        size_in, bold, colour, font = style
        body += ('<Section N=\'Character\' IX=\'0\'>'
                 '<Row IX=\'0\'>'
                 '<Cell N=\'Size\' V=\'%s\'/>'
                 '<Cell N=\'Color\' V=\'%s\'/>'
                 '<Cell N=\'Style\' V=\'%s\'/>'
                 '<Cell N=\'Font\' V=\'%s\'/>'
                 '</Row></Section>' % (num(size_in), hexcol(colour),
                                       '1' if bold else '0', esc(font)))
    body += '<Section N=\'Paragraph\' IX=\'0\'><Row IX=\'0\'><Cell N=\'HorzAlign\' V=\'%s\'/></Row></Section>' % algn

    body += extra
    body += geom if geom is not None else geom_rect(W, H)

    txt = '<Text>%s</Text>' % esc(text) if text else ''

    return ('<Shape ID=\'%d\' NameU=\'%s\' Type=\'Shape\'>'
            '%s%s</Shape>' % (sid, esc(name), body, txt))


def picture_xml(sid, name, x, y, w, h, rid):
    cx, cy = vx(x + w / 2.0), vy(y + h / 2.0)
    W, H = vx(w), (h / 72.0)
    cells = ('<Cell N=\'PinX\' V=\'%s\'/><Cell N=\'PinY\' V=\'%s\'/>'
             '<Cell N=\'Width\' V=\'%s\'/><Cell N=\'Height\' V=\'%s\'/>'
             '<Cell N=\'LocPinX\' V=\'%s\' F=\'Width*0.5\'/>'
             '<Cell N=\'LocPinY\' V=\'%s\' F=\'Height*0.5\'/>'
             '<Cell N=\'Angle\' V=\'0\'/><Cell N=\'FlipX\' V=\'0\'/><Cell N=\'FlipY\' V=\'0\'/>'
             '<Cell N=\'ResizeMode\' V=\'0\'/>'
             '<Cell N=\'ImgOffsetX\' V=\'0\' F=\'ImgWidth*0\'/>'
             '<Cell N=\'ImgOffsetY\' V=\'0\' F=\'ImgHeight*0\'/>'
             '<Cell N=\'ImgWidth\' V=\'%s\' F=\'Width*1\'/>'
             '<Cell N=\'ImgHeight\' V=\'%s\' F=\'Height*1\'/>'
             '<Cell N=\'ClippingPath\' V=\'\' E=\'#N/A\'/>'
             '<Cell N=\'TxtPinX\' V=\'%s\' F=\'Width*0.5\'/>'
             '<Cell N=\'TxtPinY\' V=\'0\' F=\'Height*0\'/>'
             '<Cell N=\'TxtWidth\' V=\'%s\' F=\'Width*1\'/>'
             '<Cell N=\'TxtHeight\' V=\'0\' F=\'Height*0\'/>'
             '<Cell N=\'TxtLocPinX\' V=\'%s\' F=\'TxtWidth*0.5\'/>'
             '<Cell N=\'TxtLocPinY\' V=\'0\' F=\'TxtHeight*0.5\'/>'
             '<Cell N=\'TxtAngle\' V=\'0\'/><Cell N=\'VerticalAlign\' V=\'0\'/>'
             % (num(cx), num(cy), num(W), num(H), num(W / 2.0), num(H / 2.0),
                num(W), num(H), num(W / 2.0), num(W), num(W / 2.0)))
    return ('<Shape ID=\'%d\' NameU=\'%s\' Type=\'Foreign\'>'
            '%s%s%s'
            '<ForeignData ForeignType=\'Bitmap\' CompressionType=\'PNG\'>'
            '<Rel r:id=\'%s\'/></ForeignData></Shape>'
            % (sid, esc(name), cells, geom_rect(W, H), '', rid))


# ------------------------------------------------------------------ build page
shapes = []
img_rels = []          # (rid, media filename)
sid = 1
img_idx = 0
counter = {'chip': 0, 'band': 0, 'sub': 0, 'line': 0, 'pic': 0, 'cap': 0}

for s in items:
    geo = re.search(r'prst="(\w+)"', s)
    g = geo.group(1) if geo else None
    box = xfrm_of(s)
    if not box:
        continue
    x, y, w, h = box
    sid += 1

    if '<p:pic>' in s:
        rid = re.search(r'r:embed="(\w+)"', s).group(1)
        fname = os.path.basename(rel_map[rid].replace('../', 'ppt/'))
        img_idx += 1
        new_rid = 'rIdImg%d' % img_idx
        img_rels.append((new_rid, fname))
        counter['pic'] += 1
        shapes.append(picture_xml(sid, 'Picture %d' % counter['pic'],
                                  x, y, w, h, new_rid))
        continue

    if g == 'line':
        counter['line'] += 1
        # Straight connector drawn as an absolute two-point polyline: the shape
        # is a plain 2-D box whose geometry holds the real endpoints, so no
        # rotation is involved and the arrowhead always lands on the true end.
        bs_x, bs_y = vx(x), vy(y)             # PPT top-left  -> Visio upper-left
        es_x, es_y = vx(x + w), vy(y + h)      # PPT bottom-right -> Visio lower-right
        bx1, by1 = min(bs_x, es_x), min(bs_y, es_y)
        bx2, by2 = max(bs_x, es_x), max(bs_y, es_y)
        bw, bh = bx2 - bx1, by2 - by1
        mx, my = (bs_x + es_x) / 2.0, (bs_y + es_y) / 2.0
        cells = ("<Cell N='PinX' V='%s'/><Cell N='PinY' V='%s'/>"
                 "<Cell N='Width' V='%s'/><Cell N='Height' V='%s'/>"
                 "<Cell N='LocPinX' V='%s' F='Width*0.5'/>"
                 "<Cell N='LocPinY' V='%s' F='Height*0.5'/>"
                 "<Cell N='Angle' V='0'/>"
                 "<Cell N='FlipX' V='0'/><Cell N='FlipY' V='0'/>"
                 "<Cell N='ResizeMode' V='0'/>"
                 "<Cell N='BeginX' V='%s'/><Cell N='BeginY' V='%s'/>"
                 "<Cell N='EndX' V='%s'/><Cell N='EndY' V='%s'/>"
                 "<Cell N='TextBkgnd' V='#ffffff'/>"
                 "<Cell N='QuickStyleLineMatrix' V='1'/>"
                 "<Cell N='QuickStyleFillMatrix' V='1'/>"
                 "<Cell N='QuickStyleEffectsMatrix' V='1'/>"
                 "<Cell N='QuickStyleFontMatrix' V='1'/>"
                 "<Cell N='LineWeight' V='%s' U='IN'/>"
                 "<Cell N='LineColor' V='%s'/>"
                 "<Cell N='LinePattern' V='1'/>"
                 "<Cell N='BeginArrow' V='0'/>"
                 "<Cell N='EndArrow' V='%s'/>"
                 "<Cell N='EndArrowSize' V='2'/>"
                 "<Cell N='EndArrowWidth' V='2'/>"
                 "<Cell N='EndArrowLength' V='2'/>"
                 % (num(mx), num(my), num(max(bw, 0.0001)), num(max(bh, 0.0001)),
                    num(max(bw, 0.0001) / 2.0), num(max(bh, 0.0001) / 2.0),
                    num(bs_x), num(bs_y), num(es_x), num(es_y),
                    num(line_width_in(s)), hexcol(line_color(s) or '000000'),
                    '13' if 'tailEnd' in s else '0'))
        shapes.append("<Shape ID='%d' NameU='Arrow %d' Type='Shape'>%s%s</Shape>"
                      % (sid, counter['line'], cells,
                         geom_line(bs_x - bx1, bs_y - by1, es_x - bx1, es_y - by1)))
        continue


    if g in ('rect', 'roundRect'):
        text = text_of(s)
        style = run_style(s)
        algn = align_of(s)
        anchor = anchor_of(s)
        fill = fill_color(s)
        lncol = line_color(s)
        lnw = line_width_in(s)

        radius = None
        if g == 'roundRect':
            am = re.search(r'<a:gd name="adj" fmla="val (\d+)"', s)
            adj = int(am.group(1)) / 100000.0 if am else 0.16667
            radius = adj * min(w, h) / 72.0

        if text.strip():
            sz = style[0] * 72.0 if style else 12
            if sz >= 13:
                counter['band'] += 1
                nm = 'Title %d' % counter['band']
            elif sz >= 9.2:
                counter['chip'] += 1
                nm = 'Step %d' % counter['chip']
            elif sz <= 7.5:
                counter['cap'] += 1
                nm = 'Caption %d' % counter['cap']
            else:
                counter['sub'] += 1
                nm = 'Sub %d' % counter['sub']
        else:
            counter['band'] += 1
            nm = 'Band %d' % counter['band']

        shapes.append(shape_xml(sid, nm, x, y, w, h, fill, lncol, lnw,
                                text, style, algn, anchor, radius))

page_shapes = ''.join(shapes)

# ------------------------------------------------------------------ package
page1 = ("<?xml version='1.0' encoding='utf-8' ?>\n"
         "<PageContents xmlns='%s' xmlns:r='%s' xml:space='preserve'>"
         "<Shapes>%s</Shapes></PageContents>" % (NS, RNS, page_shapes))

page_rels = ["<Relationship Id='rId1' Type='%s/page' Target='page1.xml'/>" % VRNS]
for rid, fname in img_rels:
    page_rels.append("<Relationship Id='%s' Type='%s/image' Target='../media/%s'/>"
                      % (rid, RNS, fname))

pages = ("<?xml version='1.0' encoding='utf-8' ?>\n"
         "<Pages xmlns='%s' xmlns:r='%s' xml:space='preserve'>"
         "<Page ID='0' NameU='Page-1' Name='流程图' ViewScale='-1' "
         "ViewCenterX='%s' ViewCenterY='%s'>"
         "<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
         "<Cell N='PageWidth' V='%s'/>"
         "<Cell N='PageHeight' V='%s'/>"
         "<Cell N='ShdwOffsetX' V='0.1181102362204724'/>"
         "<Cell N='ShdwOffsetY' V='-0.1181102362204724'/>"
         "<Cell N='PageScale' V='0.03937007874015748' U='MM'/>"
         "<Cell N='DrawingScale' V='0.03937007874015748' U='MM'/>"
         "<Cell N='DrawingSizeType' V='0'/>"
         "<Cell N='DrawingScaleType' V='0'/>"
         "<Cell N='InhibitSnap' V='0'/>"
         "<Cell N='PageLockReplace' V='0' U='BOOL'/>"
         "<Cell N='PageLockDuplicate' V='0' U='BOOL'/>"
         "<Cell N='UIVisibility' V='0'/>"
         "<Cell N='ShdwType' V='0'/><Cell N='ShdwObliqueAngle' V='0'/>"
         "<Cell N='ShdwScaleFactor' V='1'/>"
         "<Cell N='DrawingResizeType' V='1'/>"
         "<Cell N='PageShapeSplit' V='1'/>"
         "</PageSheet><Rel r:id='rId1'/></Page></Pages>"
         % (NS, RNS, num(PAGE_W_IN / 2.0), num(PAGE_H_IN / 2.0),
            num(PAGE_W_IN), num(PAGE_H_IN)))

windows = ("<?xml version='1.0' encoding='utf-8' ?>\n"
           "<Windows ClientWidth='2560' ClientHeight='1298' xmlns='%s' xmlns:r='%s' "
           "xml:space='preserve'>"
           "<Window ID='0' WindowType='Drawing' WindowState='1073741824' WindowLeft='-11' "
           "WindowTop='-45' WindowWidth='2582' WindowHeight='1354' ContainerType='Page' "
           "Page='0' ViewScale='-1' ViewCenterX='%s' ViewCenterY='%s'>"
           "<ShowRulers>1</ShowRulers><ShowGrid>1</ShowGrid><ShowPageBreaks>1</ShowPageBreaks>"
           "<ShowGuides>1</ShowGuides><ShowConnectionPoints>1</ShowConnectionPoints>"
           "<GlueSettings>9</GlueSettings><SnapSettings>65847</SnapSettings>"
           "<SnapExtensions>34</SnapExtensions><SnapAngles/>"
           "<DynamicGridEnabled>1</DynamicGridEnabled><TabSplitterPos>0.5</TabSplitterPos>"
           "</Window></Windows>" % (NS, RNS, num(PAGE_W_IN / 2.0), num(PAGE_H_IN / 2.0)))

document = ("<?xml version='1.0' encoding='utf-8' ?>\n"
            "<VisioDocument xmlns='%s' xmlns:r='%s' xml:space='preserve'>"
            "<DocumentSettings TopPage='0' DefaultTextStyle='3' DefaultLineStyle='3' "
            "DefaultFillStyle='3' DefaultGuideStyle='4'>"
            "<GlueSettings>9</GlueSettings><SnapSettings>65847</SnapSettings>"
            "<SnapExtensions>34</SnapExtensions><DynamicGridEnabled>1</DynamicGridEnabled>"
            "<ProtectStyles>0</ProtectStyles><ProtectShapes>0</ProtectShapes>"
            "<ProtectMasters>0</ProtectMasters><ProtectBkgnds>0</ProtectBkgnds>"
            "</DocumentSettings>"
            "<Colors><ColorEntry IX='0' RGB='#000000'/><ColorEntry IX='1' RGB='#FFFFFF'/></Colors>"
            "<DocumentSheet NameU='TheDoc' Name='文档' LineStyle='0' FillStyle='0' TextStyle='0'>"
            "<Cell N='OutputFormat' V='0'/><Cell N='LockPreview' V='0'/>"
            "<Cell N='AddMarkup' V='0'/><Cell N='ViewMarkup' V='0'/>"
            "<Cell N='PreviewQuality' V='0'/><Cell N='PreviewScope' V='0'/>"
            "<Cell N='DocLangID' V='2052'/>"
            "</DocumentSheet>"
            "</VisioDocument>" % (NS, RNS))

content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="xml" ContentType="application/xml"/>'
                 '<Default Extension="png" ContentType="image/png"/>'
                 '<Default Extension="emf" ContentType="image/x-emf"/>'
                 '<Override PartName="/visio/document.xml" ContentType="application/vnd.ms-visio.drawing.main+xml"/>'
                 '<Override PartName="/visio/pages/pages.xml" ContentType="application/vnd.ms-visio.pages+xml"/>'
                 '<Override PartName="/visio/pages/page1.xml" ContentType="application/vnd.ms-visio.page+xml"/>'
                 '<Override PartName="/visio/windows.xml" ContentType="application/vnd.ms-visio.windows+xml"/>'
                 '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                 '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                 '</Types>')

root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="%s/document" Target="visio/document.xml"/>'
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
             '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
             '</Relationships>' % VRNS)

doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="%s/pages" Target="pages/pages.xml"/>'
            '<Relationship Id="rId2" Type="%s/windows" Target="windows.xml"/>'
            '</Relationships>' % (VRNS, VRNS))

pages_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
              '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              '<Relationship Id="rId1" Type="%s/page" Target="page1.xml"/>'
              '</Relationships>' % VRNS)

core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>风电功率预测 项目设计流程图</dc:title>'
        '<dc:creator>WorkBuddy</dc:creator><cp:lastModifiedBy>WorkBuddy</cp:lastModifiedBy>'
        '</cp:coreProperties>')

app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
       '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
       'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
       '<Application>Microsoft Visio</Application><Pages>1</Pages>'
       '</Properties>')

out = {
    '[Content_Types].xml': content_types,
    '_rels/.rels': root_rels,
    'docProps/core.xml': core,
    'docProps/app.xml': app,
    'visio/document.xml': document,
    'visio/_rels/document.xml.rels': doc_rels,
    'visio/pages/pages.xml': pages,
    'visio/pages/_rels/pages.xml.rels': pages_rels,
    'visio/pages/page1.xml': page1,
    'visio/pages/_rels/page1.xml.rels':
        ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         + ''.join(page_rels) + '</Relationships>'),
    'visio/windows.xml': windows,
}

if os.path.exists(OUT):
    os.remove(OUT)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zo:
    for name, data in out.items():
        zo.writestr(name, data.encode('utf-8'))
    for fname, blob in media.items():
        zo.writestr('visio/media/' + fname, blob)

print('wrote', OUT, os.path.getsize(OUT), 'bytes')
print('shapes:', counter, 'images:', len(img_rels))
print('page: %.2f x %.2f in' % (PAGE_W_IN, PAGE_H_IN))
