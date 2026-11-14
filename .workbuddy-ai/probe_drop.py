# -*- coding: utf-8 -*-
"""Try VARIANT flavours for Visio Page.Drop; each attempt in-process, flush output."""
import win32com.client as win32
import pythoncom, os, glob, shutil

for d in glob.glob(os.path.join(os.environ.get('TEMP', '/tmp'), 'gen_py', '3.13', '00021A98*')):
    shutil.rmtree(d, ignore_errors=True)

app = win32.dynamic.Dispatch('Visio.Application')
app.Visible = False
doc = app.Documents.Add("")
page = doc.Pages.Item(1)
img = os.path.abspath('png/wind.png')

trials = [
    ('VT_VARIANT|VT_BYREF', lambda: win32.VARIANT(pythoncom.VT_VARIANT | pythoncom.VT_BYREF, [img])),
    ('VT_BSTR|VT_BYREF', lambda: win32.VARIANT(pythoncom.VT_BSTR | pythoncom.VT_BYREF, [img])),
    ('VT_DISPATCH|VT_BYREF', lambda: win32.VARIANT(pythoncom.VT_DISPATCH | pythoncom.VT_BYREF, [img])),
    ('VT_ARRAY|VT_VARIANT', lambda: win32.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_VARIANT, [[img]])),
]
for label, mk in trials:
    try:
        v = mk()
        p = page.Drop(v, 0.0, 0.0)
        print('%-22s OK type=%s' % (label, p.Type), flush=True)
        break
    except Exception as e:
        print('%-22s FAIL %s' % (label, repr(e)[:120]), flush=True)

doc.Close()
app.Quit()
print('DONE', flush=True)
