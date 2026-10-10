#!/usr/bin/env bash
# Fails if anything staged (or tracked) looks like a secret. Run before every push.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
files=$(git ls-files --cached --others --exclude-standard)
bad=0
# forbidden file names
if echo "$files" | grep -E '(^|/)\.env$|\.tfstate|init\.json|\.pem$|\.key$|mise\.local\.toml$|\.pt$|data/.*\.(txt|bin)$'; then
  echo "forbidden file tracked"; bad=1; fi
# token-shaped strings (placeholders like sk-ant-... or <your-key> are fine)
pattern='sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|hvs\.[A-Za-z0-9]{20,}|s\.[A-Za-z0-9]{24}$|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,}'
if echo "$files" | xargs -r grep -nIE "$pattern" -- 2>/dev/null | grep -v 'scripts/scan-secrets.sh'; then
  echo "possible secret found"; bad=1; fi
[ $bad -eq 0 ] && echo "secret scan: clean"
exit $bad
