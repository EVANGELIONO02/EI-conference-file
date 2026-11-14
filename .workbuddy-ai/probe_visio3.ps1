
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$img = "E:/学校/风电EI/png/wind.png"
$tmp = Join-Path $env:TEMP 'vs_ps_test3.vsdx'
if (Test-Path $tmp) { Remove-Item $tmp -Force }

$app = New-Object -ComObject Visio.Application
$app.Visible = $false
$doc = $app.Documents.Add("")
$page = $doc.Pages.Item(1)

Write-Output "--- A: InvokeMember with byref modifier ---"
try {
  $pm = New-Object System.Reflection.ParameterModifier 3
  $pm[0] = $true; $pm[1] = $true; $pm[2] = $true
  $args = New-Object object[] 3
  $args[0] = $img; $args[1] = 5.0; $args[2] = 5.0
  $pic = [Microsoft.VisualBasic.Interaction]::CallByName 2>$null
} catch { Write-Output "A setup fail" }

Write-Output "--- A2: type.InvokeMember ---"
try {
  $t = $page.GetType()
  $pm = New-Object System.Reflection.ParameterModifier 3
  $pm[0] = $true
  $args = [object[]]@($img, 5.0, 5.0)
  $r = $t.InvokeMember("Drop", [System.Reflection.BindingFlags]::InvokeMethod, $null, $page, $args, $pm, $null, $null)
  Write-Output "A2 OK type=$($r.Type)"
} catch { Write-Output ("A2 FAIL: " + $_.Exception.Message) }

Write-Output "--- B: clipboard image paste ---"
try {
  $bmp = [System.Drawing.Image]::FromFile($img)
  [System.Windows.Forms.Clipboard]::SetImage($bmp)
  $app.ActiveWindow.Page = $page
  $app.Selection.Paste()
  $p = $app.Selection(1)
  Write-Output "B OK type=$($p.Type)"
} catch { Write-Output ("B FAIL: " + $_.Exception.Message) }

Write-Output "--- C: page.Import ---"
try {
  $p2 = $page.Import($img)
  Write-Output "C OK type=$($p2.Type)"
} catch { Write-Output ("C FAIL: " + $_.Exception.Message) }

$doc.SaveAs($tmp)
Write-Output ("vsdx bytes: " + (Get-Item $tmp).Length)
$doc.Close()
$app.Quit()
Write-Output "DONE"
