$ErrorActionPreference = 'Stop'
$img = "E:\学校\风电EI\png\wind.png"
$tmp = Join-Path $env:TEMP 'vs_ps_test.vsdx'
$png = Join-Path $env:TEMP 'vs_ps_test.png'
if (Test-Path $tmp) { Remove-Item $tmp -Force }

$app = New-Object -ComObject Visio.Application
$app.Visible = $false
$doc = $app.Documents.Add("")
$page = $doc.Pages.Item(1)

try {
  $pic = $page.Drop($img, 5.0, 5.0)
  $pic.CellsU("LockAspect").FormulaU = "0"
  $pic.CellsU("Width").FormulaU = "2.5 in"
  $pic.CellsU("Height").FormulaU = "1.5 in"
  $pic.CellsU("PinX").FormulaU = "6.0 in"
  $pic.CellsU("PinY").FormulaU = "5.0 in"
  "picture drop OK type=$($pic.Type)"
} catch {
  "picture drop FAIL: $($_.Exception.Message)"
}

$r = $page.DrawRectangle(1.0, 1.0, 3.0, 1.4)
$r.CellsU("Rounding").FormulaU = "0.1 in"
$r.Text = "test"
"rect OK"

$doc.SaveAs($tmp)
if (Test-Path $png) { Remove-Item $png -Force }
$page.Export($png)
"vsdx bytes: " + (Get-Item $tmp).Length
"png bytes: " + (Get-Item $png).Length
$doc.Close()
$app.Quit()
"DONE"
