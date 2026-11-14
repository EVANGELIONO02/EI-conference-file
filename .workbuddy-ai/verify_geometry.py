# -*- coding: utf-8 -*-
"""Authoritative check: let Visio report every shape's real geometry, then
compare against the PPTX source in PPT (top-left) coordinates."""
import win32com.client as win32
import os, glob, shutil, zipfile, re, sys

VS = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图.vsdx'
PT = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图_修改版.pptx'
PNG = os.path.join(os.environ.get('TEMP', '/tmp'), 'vsdx_final.png')

for d in glob.glob(os.path.join(os.environ.get('TEMP', '/tmp'), 'gen_py', '3.13', '00021A98*')):
    shutil.rmtree(d, ignore_errors=True)

# ---- expected geometry from the PPTX ----
z = zipfile.ZipFile(PT)
px = z.read('ppt/slides/slide1.xml').decode('utf8')
pres = z.read('ppt/presentation.xml').decode('utf8')
m = re.search(r'<p:sldSz[^/]*cx="(\d+)"[^/]*cy="(\d+)"', pres)
PAGE_H = int(m.group(2)) / 12700.0
body = px[px.find('<p:spTree>'):]
ppt = []
for mm in re.finditer(r'<p:(?:sp|pic)>(?:(?!</p:(?:sp|pic)>).)*</p:(?:sp|pic)>', body, re.S):
    s = mm.group(0)
    o = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"/><a:ext cx="(\d+)" cy="(\d+)"', s)
    x, y, w, h = [int(v) / 12700.0 for v in o.groups()]
    t = ''.join(re.findall(r'<a:t>([^<]*)</a:t>', s))
    is_line = 'prst="line"' in s
    ppt.append(dict(x=x, y=y, w=w, h=h, t=t.strip(), line=is_line))

# ---- actual geometry as Visio sees it ----
app = win32.dynamic.Dispatch('Visio.Application')
app.Visible = False
doc = app.Documents.Open(os.path.abspath(VS))
page = doc.Pages.Item(1)
print('Visio opened the file. shapes =', page.Shapes.Count, flush=True)

actual = []
for sh in page.Shapes:
    try:
        txt = sh.Text.strip()
    except Exception:
        txt = ''
    # a Visio connector is a 1-D shape: its Height is 0 and it has Begin/End
    try:
        hgt = sh.CellsU('Height').ResultIU
        wid = sh.CellsU('Width').ResultIU
    except Exception:
        hgt, wid = 0.0, 0.0
    try:
        pic = (sh.Type == 4)
    except Exception:
        pic = False
    if pic:
        x = sh.CellsU('PinX').ResultIU * 72 - wid * 36
        y = PAGE_H - (sh.CellsU('PinY').ResultIU * 72 + hgt * 36)
        actual.append(dict(kind='pic', x=x, y=y, w=wid * 72, h=hgt * 72, t=txt))
        continue
    if abs(hgt) < 1e-6 and abs(wid) > 1e-6:
        # 1-D connector: read its real endpoints
        bx = sh.CellsU('BeginX').ResultIU * 72
        by = sh.CellsU('BeginY').ResultIU * 72
        ex = sh.CellsU('EndX').ResultIU * 72
        ey = sh.CellsU('EndY').ResultIU * 72
        actual.append(dict(kind='line', x1=bx, y1=PAGE_H - by, x2=ex, y2=PAGE_H - ey, t=txt))
        continue
    x = sh.CellsU('PinX').ResultIU * 72 - wid * 36
    y = PAGE_H - (sh.CellsU('PinY').ResultIU * 72 + hgt * 36)
    actual.append(dict(kind='shape', x=x, y=y, w=wid * 72, h=hgt * 72, t=txt))

try:
    page.Export(PNG)
except Exception as e:
    print('export fail', e)

# ---- compare ----
def close(a, b, tol=1.5):
    return abs(a - b) <= tol

problems = []
# labelled shapes by text
ppt_by_text = {}
for p in ppt:
    if p['t']:
        ppt_by_text.setdefault(p['t'], p)
seen = set()
for a in actual:
    if a['kind'] != 'shape' or not a['t']:
        continue
    p = ppt_by_text.get(a['t'])
    if not p:
        continue
    seen.add(a['t'])
    if not (close(a['x'], p['x']) and close(a['y'], p['y'])
            and close(a['w'], p['w']) and close(a['h'], p['h'])):
        problems.append(('shape', a['t'][:24], (p['x'], p['y'], p['w'], p['h']),
                         (a['x'], a['y'], a['w'], a['h'])))

# pictures
ppt_pics = [p for p in ppt if p['w'] > 100 and p['h'] > 80 and p['x'] >= 200]
v_pics = [a for a in actual if a['kind'] == 'pic']
for p in ppt_pics:
    best = min(v_pics, key=lambda a: abs(a['x'] - p['x']) + abs(a['y'] - p['y'])) if v_pics else None
    if best and not (close(best['x'], p['x']) and close(best['y'], p['y'])
                      and close(best['w'], p['w']) and close(best['h'], p['h'])):
        problems.append(('pic', (p['x'], p['y'], p['w'], p['h']),
                         (best['x'], best['y'], best['w'], best['h'])))

# lines: compare as unordered endpoint sets
def lk(p):
    return (round(min(p['x'], p['x'] + p['w'])), round(min(p['y'], p['y'] + p['h'])),
            round(max(p['x'], p['x'] + p['w'])), round(max(p['y'], p['y'] + p['h'])))

ppt_lines = sorted(lk(p) for p in ppt if p['line'])
vis_lines = sorted((round(min(a['x1'], a['x2'])), round(min(a['y1'], a['y2'])),
                    round(max(a['x1'], a['x2'])), round(max(a['y1'], a['y2']))) for a in actual if a['kind'] == 'line')
print('ppt lines', len(ppt_lines), 'visio lines', len(vis_lines))
for i, (a, b) in enumerate(zip(ppt_lines, vis_lines)):
    if not all(close(u, v, 1.0) for u, v in zip(a, b)):
        problems.append(('line', a, b))

print('\n=== PROBLEMS ===')
for pr in problems:
    print(' ', pr)
if not problems:
    print('  none - VSDX geometry matches the PPTX exactly')

print('\nPNG:', PNG, os.path.exists(PNG))
try:
    doc.Close(); app.Quit()
except Exception:
    pass
