# -*- coding: utf-8 -*-
"""Check: can Visio still SaveAs? Can it open the project's existing vsdx?"""
import win32com.client as win32
import os, glob, shutil

for d in glob.glob(os.path.join(os.environ.get('TEMP', '/tmp'), 'gen_py', '3.13', '00021A98*')):
    shutil.rmtree(d, ignore_errors=True)

app = win32.dynamic.Dispatch('Visio.Application')
app.Visible = False
doc = app.Documents.Add("")
page = doc.Pages.Item(1)
r = page.DrawRectangle(1, 1, 3, 1.5)
r.Text = 'Hello'
print('draw OK', flush=True)

out = os.path.join(os.environ.get('TEMP', '/tmp'), 'visio_ref2.vsdx')
if os.path.exists(out):
    os.remove(out)
try:
    doc.SaveAs(out)
    print('SaveAs OK', os.path.getsize(out), flush=True)
except Exception as e:
    print('SaveAs FAIL', repr(e)[:200], flush=True)

try:
    doc.Close()
    app.Quit()
except Exception as e:
    print('close fail', repr(e)[:100], flush=True)

# now try opening the project's existing vsdx
print('--- open existing project vsdx ---', flush=True)
app2 = win32.dynamic.Dispatch('Visio.Application')
app2.Visible = False
try:
    d2 = app2.Documents.Open(os.path.abspath('visio/workflow.vsdx'))
    p2 = d2.Pages.Item(1)
    print('OPEN OK, shapes =', p2.Shapes.Count, flush=True)
    d2.Close()
except Exception as e:
    print('OPEN FAIL', repr(e)[:200], flush=True)
try:
    app2.Quit()
except Exception:
    pass
print('DONE', flush=True)
