# -*- coding: utf-8 -*-
"""Let Visio author a line with an arrowhead, then read back the XML it wrote."""
import win32com.client as win32
import os, glob, shutil, zipfile, re

for d in glob.glob(os.path.join(os.environ.get('TEMP', '/tmp'), 'gen_py', '3.13', '00021A98*')):
    shutil.rmtree(d, ignore_errors=True)

out = os.path.join(os.environ.get('TEMP', '/tmp'), 'visio_line.vsdx')
if os.path.exists(out):
    os.remove(out)

app = win32.dynamic.Dispatch('Visio.Application')
app.Visible = False
doc = app.Documents.Add("")
page = doc.Pages.Item(1)

# horizontal line with arrow
ln = page.DrawLine(1.0, 5.0, 4.0, 5.0)
ln.CellsU('EndArrow').FormulaU = '13'
ln.CellsU('LineWeight').FormulaU = '1.5 pt'
ln.CellsU('LineColor').FormulaU = 'RGB(47,143,122)'
print('horizontal line drawn', flush=True)

# vertical line with arrow
ln2 = page.DrawLine(1.0, 1.0, 1.0, 4.0)
ln2.CellsU('EndArrow').FormulaU = '13'
print('vertical line drawn', flush=True)

doc.SaveAs(out)
print('saved', os.path.getsize(out), flush=True)
try:
    doc.Close(); app.Quit()
except Exception:
    pass

z = zipfile.ZipFile(out)
x = z.read('visio/pages/page1.xml').decode('utf8')
print('\n=== page1.xml ===')
print(x)
