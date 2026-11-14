
$ErrorActionPreference = 'Continue'
$out = "C:/Users/ROG/AppData/Local/Temp/visio_ref.vsdx"
if (Test-Path $out) { Remove-Item $out -Force }
$app = New-Object -ComObject Visio.Application
$app.Visible = $false
$doc = $app.Documents.Add("")
$page = $doc.Pages.Item(1)
$r = $page.DrawRectangle(1,1,3,1.5)
$r.Text = "Hello"
$l = $page.DrawLine(3,1.25,5,1.25)
$l.CellsU("EndArrow").FormulaU = "4"
try { $doc.SaveAs($out); Write-Output ("saved " + (Get-Item $out).Length) }
catch { Write-Output ("SaveAs FAIL: " + $_.Exception.Message) }
try { $doc.SaveAs($out) } catch { Write-Output ("SaveAs2 FAIL: " + $_.Exception.Message) }
try { $doc.Save() } catch { Write-Output ("Save FAIL: " + $_.Exception.Message) }
try { $doc.Close() } catch { Write-Output ("Close FAIL: " + $_.Exception.Message) }
try { $app.Quit() } catch { Write-Output ("Quit FAIL: " + $_.Exception.Message) }
Write-Output "DONE"
