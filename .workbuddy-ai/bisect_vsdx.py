# -*- coding: utf-8 -*-
"""Binary search which package part makes Visio reject the file.

Strategy: start from the Visio-authored reference package (known good),
swap in my parts one at a time, and test each candidate with Visio.
"""
import zipfile, os, shutil, glob, sys
import win32com.client as win32

REF = os.path.join(os.environ.get('TEMP', '/tmp'), 'visio_ref2.vsdx')
MINE = '基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图.vsdx'
TMP = os.path.join(os.environ.get('TEMP', '/tmp'), 'vsdx_bisect.vsdx')

for d in glob.glob(os.path.join(os.environ.get('TEMP', '/tmp'), 'gen_py', '3.13', '00021A98*')):
    shutil.rmtree(d, ignore_errors=True)

ref = zipfile.ZipFile(REF)
mine = zipfile.ZipFile(MINE)
ref_parts = {n: ref.read(n) for n in ref.namelist()}
mine_parts = {n: mine.read(n) for n in mine.namelist()}
ref.close(); mine.close()

app = win32.dynamic.Dispatch('Visio.Application')
app.Visible = False


def test(parts, label):
    if os.path.exists(TMP):
        os.remove(TMP)
    with zipfile.ZipFile(TMP, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, d in parts.items():
            z.writestr(n, d)
    try:
        doc = app.Documents.Open(TMP)
        cnt = doc.Pages.Item(1).Shapes.Count
        try:
            doc.Close()
        except Exception:
            pass
        print('%-58s OPEN OK (shapes=%s)' % (label, cnt), flush=True)
        return True
    except Exception as e:
        print('%-58s FAIL' % label, flush=True)
        return False


# baseline
assert test(dict(ref_parts), 'baseline (ref only)')

# my parts that exist in ref
common = [n for n in mine_parts if n in ref_parts]
for n in common:
    p = dict(ref_parts)
    p[n] = mine_parts[n]
    test(p, 'swap ' + n)

# my extra parts (add to ref)
for n in mine_parts:
    if n not in ref_parts:
        p = dict(ref_parts)
        p[n] = mine_parts[n]
        # also add rels/content-type entries? just test raw
        test(p, 'ADD  ' + n)

try:
    app.Quit()
except Exception:
    pass
print('DONE', flush=True)
