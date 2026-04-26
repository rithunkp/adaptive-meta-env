param(
    [string]$SpaceApp = "https://itzrick-openadapt.hf.space",
    [int]$Tail = 2000
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "artifacts/metrics"
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$statusPath = Join-Path $outDir "space_training_status_$timestamp.json"
$logsPath = Join-Path $outDir "space_training_logs_$timestamp.json"
$policyPath = Join-Path $outDir "space_training_policy_$timestamp.json"

$status = Invoke-RestMethod -Uri "$SpaceApp/training/status" -Method Get
$logs = Invoke-RestMethod -Uri "$SpaceApp/training/logs?tail=$Tail" -Method Get
$policy = Invoke-RestMethod -Uri "$SpaceApp/training/policy" -Method Get

$status | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $statusPath -Encoding UTF8
$logs | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $logsPath -Encoding UTF8
$policy | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath $policyPath -Encoding UTF8

Write-Output "Saved:"
Write-Output " - $statusPath"
Write-Output " - $logsPath"
Write-Output " - $policyPath"
