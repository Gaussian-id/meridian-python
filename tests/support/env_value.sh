#!/usr/bin/env bash
# The engine's local-up reader, run over test fixtures only. env_value below is copied verbatim from
# `1. Engine/scripts/local-common.sh` lines 125-132 (engine commit cd572cf0); local-up.sh leaves simulator mode when
# it prints a non-empty value for either paid key. Same shell options as local-up.sh.
#   bash env_value.sh <folder>
# Prints `<file stem><TAB><0|1><TAB><0|1>` per *.env file in the folder: whether the Gemini and the OpenRouter key
# come back non-empty. Never a value.
set -euo pipefail

env_value() {
  local line
  line=$(grep -E "^$1=" "$2" | head -n 1 | tr -d '\r' || true)
  line=${line#*=}
  line=${line#\"}
  line=${line%\"}
  printf '%s' "$line"
}

for file in "$1"/*.env; do
  gemini=0
  openrouter=0
  if [ -n "$(env_value GAUSSMERIDIAN_GEMINI_API_KEY "$file")" ]; then gemini=1; fi
  if [ -n "$(env_value GAUSSMERIDIAN_OPENROUTER_API_KEY "$file")" ]; then openrouter=1; fi
  stem=${file##*/}
  printf '%s\t%s\t%s\n' "${stem%.env}" "$gemini" "$openrouter"
done
