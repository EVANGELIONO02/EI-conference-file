
$ErrorActionPreference = 'Stop'
$vsdx = "E:/学校/风电EI/基于气象信息引导与条件扩散模型的风电功率预测方法_项目设计流程图.vsdx"
$png  = "C:/Users/ROG/AppData/Local/Temp/vsdx_render.png"
$app = New-Object -ComObject Visio.Application
$app.Visible = $false
try {
  $doc = $app.Documents.Open($vsdx)
  $page = $doc.Pages.Item(1)
  Write-Output ("shapes on page: " + $page.Shapes.Count)
  $n=0
  foreach ($s in $page.Shapes) {
    $n++
    if ($n -le 8) {
      $tx = ""
      try { $tx = $s.Text } catch {}
      Write-Output ("  " + $n + " type=" + $s.Type + " name=" + $s.NameU + " text=" + $tx)
    }
  }
  if (Test-Path $png) { Remove-Item $png -Force }
  $page.Export($png)
  Write-Output ("png bytes: " + (Get-Item $png).Length)
  $doc.Close()
} catch {
  Write-Output ("OPEN FAIL: " + $_.Exception.Message)
}
$app.Quit()
Write-Output "DONE"
