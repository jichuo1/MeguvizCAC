<#
.SYNOPSIS
  MeguvizCAC v0.8.0 External Gemini Data Factory — Windows 1-Click Launcher (Zero Antigravity Quota Cost!)
.EXAMPLE
  # 1) Zero-cost local self-test (no API key needed):
  .\tools\gemini_v08_datafactory\run_factory.ps1 -Mode dry-run -LimitA 100 -LimitB 100 -LimitC 40 -LimitD 60

  # 2) Full 250k production run using Gemini Batch API (50% discount!):
  $env:GEMINI_API_KEY="AIza..."
  .\tools\gemini_v08_datafactory\run_factory.ps1 -Mode batch -SubmitBatch -PollBatch

  # 3) Fast 100k realtime concurrent run:
  $env:GEMINI_API_KEY="AIza..."
  .\tools\gemini_v08_datafactory\run_factory.ps1 -Mode realtime -LimitA 30000 -LimitB 45000 -LimitC 12000 -LimitD 13000 -Workers 12 -Qps 8
#>
param(
    [ValidateSet("batch", "realtime", "pack", "dry-run")]
    [string]$Mode = "batch",
    [string]$WorkDir = "v08-datafactory-work",
    [int]$LimitA = 60000,
    [int]$LimitB = 120000,
    [int]$LimitC = 30000,
    [int]$LimitD = 40000,
    [int]$BatchSize = 25,
    [int]$Workers = 8,
    [double]$Qps = 6.0,
    [string]$Tier1Model = "gemini-2.5-flash",
    [string]$Tier2Model = "gemini-2.5-pro",
    [switch]$SubmitBatch,
    [switch]$PollBatch,
    [switch]$SkipTier2
)

$ErrorActionPreference = "Stop"
function Invoke-Checked {
    param([string[]]$CmdArgs)
    & python @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : python $($CmdArgs -join ' ')" }
}

Write-Host "=== [1/4] Exporting Zero-PII Anti-Leakage Queues (A/B/C/D) -> $WorkDir ===" -ForegroundColor Cyan
Invoke-Checked @(
    "tools/gemini_v08_datafactory/01_export_queue.py",
    "--out-dir", $WorkDir,
    "--limit-a", $LimitA, "--limit-b", $LimitB, "--limit-c", $LimitC, "--limit-d", $LimitD,
    "--batch-size", $BatchSize
)

$batchFlags = @()
if ($SubmitBatch) { $batchFlags += "--submit-batch" }
if ($PollBatch) { $batchFlags += "--poll-batch" }

Write-Host "`n=== [2/4] Running Tier-1 ($Tier1Model) [mode=$Mode] ===" -ForegroundColor Cyan
Invoke-Checked (@(
    "tools/gemini_v08_datafactory/02_run_gemini_factory.py",
    "--work-dir", $WorkDir, "--mode", $Mode, "--tier", "tier1", "--model", $Tier1Model,
    "--workers", $Workers, "--qps", $Qps
) + $batchFlags)

if (-not $SkipTier2 -and ($Mode -eq "realtime" -or $Mode -eq "dry-run" -or ($Mode -eq "batch" -and $PollBatch))) {
    Write-Host "`n=== [3/4] Escalating Ambiguous/Trap/Disagreement Rows -> Tier-2 ($Tier2Model) ===" -ForegroundColor Cyan
    Invoke-Checked (@(
        "tools/gemini_v08_datafactory/02_run_gemini_factory.py",
        "--work-dir", $WorkDir, "--mode", $Mode, "--tier", "tier2", "--model", $Tier2Model,
        "--workers", ([Math]::Max(2, [int]($Workers / 2))), "--qps", ([Math]::Max(1.0, $Qps / 2.0))
    ) + $batchFlags)
}

if ($Mode -eq "realtime" -or $Mode -eq "dry-run" -or ($Mode -eq "batch" -and $PollBatch)) {
    Write-Host "`n=== [4/4] Hidden Canary QA Gate + Anti-Leakage Re-Audit + Release Export ===" -ForegroundColor Cyan
    Invoke-Checked @("tools/gemini_v08_datafactory/03_qc_and_merge.py", "--work-dir", $WorkDir)
    Write-Host "`n[DONE] Clean v0.8.0 outputs saved under $WorkDir\release\" -ForegroundColor Green
} else {
    Write-Host "`n[PAUSED] Generated external artifacts under $WorkDir\. Complete external Batch/Pack run, then run Step 3 (Tier-2) & Step 4 (03_qc_and_merge.py)." -ForegroundColor Yellow
}
