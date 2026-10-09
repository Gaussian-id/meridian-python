# The engine's local-up reader, run over test fixtures only. Get-EnvValue below is copied verbatim from
# `1. Engine/scripts/local-common.ps1` lines 112-123 (engine commit cd572cf0); local-up.ps1 leaves simulator mode
# when it returns a non-empty value for either paid key. Default culture, as local-up runs it.
#   pwsh -NoProfile -NonInteractive -File get_env_value.ps1 <folder>
# Prints `<file stem><TAB><0|1><TAB><0|1>` per *.env file in the folder: whether the Gemini and the OpenRouter key
# come back non-empty. Never a value.
param([string]$Dir)
$ErrorActionPreference = 'Stop'

function Get-EnvValue([string]$Name, [string]$Path) {
    foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
        if ($line.StartsWith("$Name=")) {
            $value = $line.Substring($Name.Length + 1)
            if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }
    return ''
}

foreach ($file in Get-ChildItem -LiteralPath $Dir -Filter '*.env' | Sort-Object Name) {
    $gemini = [int][bool](Get-EnvValue 'GAUSSMERIDIAN_GEMINI_API_KEY' $file.FullName)
    $openrouter = [int][bool](Get-EnvValue 'GAUSSMERIDIAN_OPENROUTER_API_KEY' $file.FullName)
    "$($file.BaseName)`t$gemini`t$openrouter"
}
