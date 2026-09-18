$ErrorActionPreference = 'Stop'
$dst = Join-Path $env:USERPROFILE 'AiriChat'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$src = '\\wsl.localhost\Ubuntu\home\fengduan\kokoro-tts\local_chat'
Copy-Item -LiteralPath (Join-Path $src 'start_airi.bat') -Destination $dst -Force
Copy-Item -LiteralPath (Join-Path $src 'stop_airi.bat')  -Destination $dst -Force
Copy-Item -LiteralPath (Join-Path $src 'assets\airi.ico') -Destination $dst -Force
Write-Output ("files copied to: " + $dst)

$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell

$bat = Join-Path $dst 'start_airi.bat'
$lnk = $ws.CreateShortcut((Join-Path $desktop '艾莉语音聊天.lnk'))
$lnk.TargetPath = "$env:SystemRoot\System32\cmd.exe"
$lnk.Arguments = '/c "' + $bat + '"'
$lnk.WorkingDirectory = $dst
$lnk.IconLocation = (Join-Path $dst 'airi.ico')
$lnk.Description = 'Airi local chat: LLM + Kokoro TTS + RVC voice'
$lnk.Save()

$bat2 = Join-Path $dst 'stop_airi.bat'
$lnk2 = $ws.CreateShortcut((Join-Path $desktop '停止艾莉语音.lnk'))
$lnk2.TargetPath = "$env:SystemRoot\System32\cmd.exe"
$lnk2.Arguments = '/c "' + $bat2 + '"'
$lnk2.WorkingDirectory = $dst
$lnk2.IconLocation = (Join-Path $dst 'airi.ico')
$lnk2.Save()

Write-Output "---- desktop shortcuts ----"
Get-ChildItem $desktop -Filter '*.lnk' | Where-Object { $_.Name -match '艾莉' } |
  ForEach-Object { Write-Output ($_.Name + "  ->  " + $ws.CreateShortcut($_.FullName).TargetPath + " " + $ws.CreateShortcut($_.FullName).Arguments) }
