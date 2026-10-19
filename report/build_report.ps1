param(
    [string]$CommitHash = ""
)

$ErrorActionPreference = "Stop"
$reportRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$assignmentRoot = Split-Path -Parent $reportRoot
$templatePath = Join-Path $reportRoot "report.html"
$renderedPath = Join-Path $reportRoot "_rendered_report.html"
$pdfPath = Join-Path $reportRoot "final_report.pdf"
$profilePath = Join-Path $reportRoot ".chrome-profile"

if ([string]::IsNullOrWhiteSpace($CommitHash)) {
    $CommitHash = (& git -C $assignmentRoot rev-parse HEAD).Trim()
}
if ($CommitHash -notmatch '^[0-9a-fA-F]{7,40}$') {
    throw "CommitHash must contain 7 to 40 hexadecimal characters."
}

$chromeCandidates = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)
$browser = $chromeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $browser) {
    throw "Chrome or Edge was not found."
}

$html = Get-Content -LiteralPath $templatePath -Raw -Encoding UTF8
$html = $html.Replace("{{COMMIT_HASH}}", $CommitHash)
Set-Content -LiteralPath $renderedPath -Value $html -Encoding UTF8
if (Test-Path -LiteralPath $pdfPath) {
    Remove-Item -LiteralPath $pdfPath -Force
}

$renderedUri = [System.Uri]::new($renderedPath).AbsoluteUri
& $browser `
    --headless=new `
    --disable-gpu `
    --allow-file-access-from-files `
    --no-pdf-header-footer `
    --user-data-dir="$profilePath" `
    --print-to-pdf="$pdfPath" `
    $renderedUri

$deadline = [DateTime]::UtcNow.AddSeconds(15)
while (-not (Test-Path -LiteralPath $pdfPath) -and [DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds 250
}
if (-not (Test-Path -LiteralPath $pdfPath) -or (Get-Item -LiteralPath $pdfPath).Length -eq 0) {
    throw "The report PDF was not created."
}

Write-Output "Created $pdfPath"
