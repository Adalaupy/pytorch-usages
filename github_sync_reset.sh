#!/usr/bin/env zsh

git fetch origin
git reset --hard origin/main
git clean -fdx -e myenv -e .env -e github_sync_reset.sh
git status --short --branch
git ls-files --others --exclude-standard
git ls-files --others -i --exclude-standard | rg '^(myenv/|\.env$|github_sync_reset\.sh$)' -n || true