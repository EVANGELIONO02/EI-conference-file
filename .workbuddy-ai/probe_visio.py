# -*- coding: utf-8 -*-
"""Probe Visio COM capabilities needed to rebuild the slide as native editable shapes."""
import win32com.client as win32
import os, traceback

app = win32.Dispatch('Visio.Application')      # dynamic binding: avoids gencache quirks
app.Visible = False
doc = app.Documents.Add("")
page = doc.Pages.Item(1)


def cell(sh, name, formula=None, result=False):
    c = sh.CellsU(name)
    if formula is not None:
        c.FormulaU = formula
    return c.ResultIU if result else c.FormulaU


print('masters count:', doc.Masters.Count)
names = []
for i in range(1, doc.Masters.Count + 1):
    names.append(doc.Masters.Item(i).NameU)
print('master NameU:', names[:30])

try:
    m = doc.Masters.ItemU('Rounded Rectangle')
    print('MASTER rounded rectangle: OK')
except Exception as e:
    print('MASTER rounded rectangle: FAIL', repr(e)[:120])

try:
    sh = page.DrawRectangle(1.0, 1.0, 2.0, 1.4)
    sh.CellsU('Rounding').FormulaU = '0.1 in'
    print('Rounding cell on rectangle: OK ->', sh.CellsU('Rounding').ResultIU)
except Exception as e:
    print('Rounding cell on rectangle: FAIL', repr(e)[:120])

try:
    ln = page.DrawLine(3.0, 1.0, 4.0, 1.0)
    ln.CellsU('EndArrow').FormulaU = '4'
    print('EndArrow=4: OK')
except Exception as e:
    print('EndArrow: FAIL', repr(e)[:120])

try:
    t = page.DrawRectangle(5.0, 1.0, 6.0, 1.4)
    t.Text = '测试中文'
    t.CellsU('Char.Size').FormulaU = '9.5 pt'
    t.CellsU('Char.Font').FormulaU = '"微软雅黑"'
    print('Char.Font by name: OK ->', t.CellsU('Char.Font').FormulaU)
except Exception as e:
    print('Char.Font by name: FAIL', repr(e)[:120])

try:
    t2 = page.DrawRectangle(7.0, 1.0, 8.0, 1.4)
    t2.Text = 'abc'
    t2.CellsU('Para.HorzAlign').FormulaU = '1'
    t2.CellsU('VerticalAlign').FormulaU = '1'
    t2.CellsU('FillPattern').FormulaU = '0'
    t2.CellsU('LinePattern').FormulaU = '0'
    print('align/no-fill: OK')
except Exception as e:
    print('align: FAIL', repr(e)[:120])

try:
    img = os.path.abspath('png/wind.png')
    pic = page.Drop(img)
    pic.CellsU('Width').FormulaU = '1.5 in'
    pic.CellsU('Height').FormulaU = '0.9 in'
    print('Drop picture: OK, type=', pic.Type)
except Exception as e:
    print('Drop picture: FAIL', repr(e)[:120])
    traceback.print_exc()

try:
    out = os.path.abspath(os.path.join(os.environ.get('TEMP', '/tmp'), 'probe.png'))
    if os.path.exists(out):
        os.remove(out)
    page.Export(out)
    print('Export PNG: OK', os.path.exists(out))
except Exception as e:
    print('Export: FAIL', repr(e)[:120])

vsdx = os.path.abspath(os.path.join(os.environ.get('TEMP', '/tmp'), 'probe.vsdx'))
try:
    doc.SaveAs(vsdx)
    print('SaveAs vsdx: OK', os.path.getsize(vsdx))
except Exception as e:
    print('SaveAs: FAIL', repr(e)[:120])

doc.Close()
app.Quit()
print('DONE')
