param(
    [string[]]$Sequences = @('03', '06', '05', '07', '10', '09', '00', '02', '08', '01', '04'),
    [int]$MaxFrames = 0,
    [string]$OutputFolder = 'results\benchmark-batch',
    [string]$LogName = 'batch.log',
    [switch]$GraphFromRaw,
    [hashtable]$TrackedPairs = @{},
    [switch]$ForceRetest
)
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$scratchRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.datasets\batch-scratch'))
$outputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputFolder))
if (-not $outputRoot.StartsWith($repoRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay inside the workspace' }
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MPLCONFIGDIR = Join-Path $repoRoot '.mpl-cache'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$logPath = [IO.Path]::GetFullPath((Join-Path $outputRoot $LogName))
if (-not $logPath.StartsWith($outputRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Log path must stay inside result folder' }
foreach ($sequence in $Sequences) {
    if ($sequence -notmatch '^(0[0-9]|10)$') { throw "Invalid ground-truth sequence: $sequence" }
    $target = [IO.Path]::GetFullPath((Join-Path $scratchRoot $sequence))
    if (-not $target.StartsWith($scratchRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Scratch path escapes owned directory' }
    $argsForRun = @((Join-Path $PSScriptRoot 'run_kitti_stream.py'), '--sequence', $sequence, '--output-root', $outputRoot)
    if ($MaxFrames -gt 0) { $argsForRun += @('--max-frames', "$MaxFrames") }
    if ($ForceRetest) { $argsForRun += '--force-retest' }
    if ($GraphFromRaw) {
        if (-not $TrackedPairs.ContainsKey($sequence)) { throw 'Graph recovery needs tracking counts verified from the original log' }
        $argsForRun += @('--graph-from-raw', '--tracked-pairs', "$($TrackedPairs[$sequence])")
    }
    "START $sequence $(Get-Date -Format o)" | Tee-Object -FilePath $logPath -Append
    # A fresh process releases native image/descriptor memory between sequences.
    $ErrorActionPreference = 'Continue'
    & $python @argsForRun 2>&1 | Tee-Object -FilePath $logPath -Append
    $resultCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    "END $sequence exit=$resultCode $(Get-Date -Format o)" | Tee-Object -FilePath $logPath -Append
    if (Test-Path -LiteralPath $target) {
        $resolved = (Resolve-Path -LiteralPath $target).Path
        if ($resolved -ne $target -or -not $resolved.StartsWith($scratchRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected resolved cleanup path' }
        if ((Get-Item -LiteralPath $target).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing reparse-point cleanup' }
        $owner = Get-Content -LiteralPath (Join-Path $target 'owner.json') -Raw | ConvertFrom-Json
        if ($owner.sequence -ne $sequence -or $owner.purpose -ne 'temporary KITTI benchmark images and geometry') { throw 'Unrecognized scratch owner' }
        $links = Get-ChildItem -LiteralPath $target -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }
        if ($links) { throw 'Refusing cleanup containing reparse points' }
        Remove-Item -LiteralPath $target -Recurse -Force
        "CLEANED owned scratch $sequence; retained reports, trajectories and constraints" | Tee-Object -FilePath $logPath -Append
    }
    & $python (Join-Path $PSScriptRoot 'update_dashboard_benchmarks.py') --data-root (Join-Path $repoRoot 'results\benchmark-batch\reference')
    & $python (Join-Path $PSScriptRoot 'summarize_kitti_batch.py') --output-root $outputRoot
}
