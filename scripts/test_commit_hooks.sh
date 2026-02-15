#!/usr/bin/env bash
#
# test_commit_hooks.sh - Table-driven tests for commit message
# normalization and validation hooks.
#
# Tests the same transforms used by:
#   .githooks/prepare-commit-msg  (normalization)
#   .githooks/commit-msg          (validation)
#   .github/workflows/commit-lint.yml (CI normalization + validation)
#
# Usage: scripts/test_commit_hooks.sh

set -euo pipefail

PASS=0
FAIL=0
ERRORS=""

# ---------- helpers ----------

normalize() {
  local msg="$1"

  # 1. Lowercase type prefix
  msg=$(printf '%s\n' "$msg" | awk '{
    match($0, /^[A-Za-z]+/)
    if (RSTART == 1) {
      prefix = substr($0, 1, RLENGTH)
      rest = substr($0, RLENGTH + 1)
      print tolower(prefix) rest
    } else {
      print
    }
  }')

  # 2. Expand type aliases
  msg=$(printf '%s\n' "$msg" | sed -E 's/^feature([(:!])/feat\1/')
  msg=$(printf '%s\n' "$msg" | sed -E 's/^(bugfix|hotfix)([(:!])/fix\2/')
  msg=$(printf '%s\n' "$msg" | sed -E 's/^doc([(:!])/docs\1/')

  # 3. Fix missing space after colon
  msg=$(printf '%s\n' "$msg" | sed -E 's/^([a-z]+(\([a-zA-Z0-9_/ -]+\))?!?:)([^ ])/\1 \3/')

  printf '%s' "$msg"
}

validate() {
  local msg="$1"
  local TYPES="feat|fix|perf|docs|chore|ci|test|refactor|style|build|revert"

  if ! printf '%s\n' "$msg" | grep -qE "^($TYPES)(\([a-zA-Z0-9_/ -]+\))?!?: .+"; then
    return 1
  fi

  local len=${#msg}
  if [ "$len" -gt 72 ]; then
    return 1
  fi

  return 0
}

is_exempt() {
  local msg="$1"
  case "$msg" in
    Merge\ *|fixup!\ *|squash!\ *|bd\ sync*)
      return 0
      ;;
  esac
  return 1
}

assert_normalize() {
  local input="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual=$(normalize "$input")

  if [ "$actual" = "$expected" ]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}FAIL [normalize] ${label}\n  input:    '${input}'\n  expected: '${expected}'\n  actual:   '${actual}'\n\n"
  fi
}

assert_valid() {
  local msg="$1"
  local label="$2"

  if validate "$msg"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}FAIL [valid] ${label}\n  msg: '${msg}'\n  expected valid but got invalid\n\n"
  fi
}

assert_invalid() {
  local msg="$1"
  local label="$2"

  if validate "$msg"; then
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}FAIL [invalid] ${label}\n  msg: '${msg}'\n  expected invalid but got valid\n\n"
  else
    PASS=$((PASS + 1))
  fi
}

assert_exempt() {
  local msg="$1"
  local label="$2"

  if is_exempt "$msg"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}FAIL [exempt] ${label}\n  msg: '${msg}'\n  expected exempt but was not\n\n"
  fi
}

assert_not_exempt() {
  local msg="$1"
  local label="$2"

  if is_exempt "$msg"; then
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}FAIL [not-exempt] ${label}\n  msg: '${msg}'\n  expected not exempt but was exempt\n\n"
  else
    PASS=$((PASS + 1))
  fi
}

# ---------- Normalization tests ----------

# 1. Lowercase type prefix
assert_normalize "Feat: add thing"         "feat: add thing"         "uppercase Feat"
assert_normalize "CHORE: update deps"      "chore: update deps"      "all-caps CHORE"
assert_normalize "FIX(scope): bug"         "fix(scope): bug"         "uppercase FIX with scope"
assert_normalize "feat: already lowercase"  "feat: already lowercase"  "already lowercase (noop)"
assert_normalize "Docs!: breaking"         "docs!: breaking"         "uppercase with bang"

# 2. Expand type aliases
assert_normalize "feature: new thing"      "feat: new thing"         "feature -> feat"
assert_normalize "feature(ui): new thing"  "feat(ui): new thing"     "feature(scope) -> feat(scope)"
assert_normalize "bugfix: oops"            "fix: oops"               "bugfix -> fix"
assert_normalize "hotfix: urgent"          "fix: urgent"             "hotfix -> fix"
assert_normalize "doc: update readme"      "docs: update readme"     "doc -> docs"
assert_normalize "doc(api): endpoints"     "docs(api): endpoints"    "doc(scope) -> docs(scope)"

# 3. Fix missing space after colon
assert_normalize "feat:add thing"          "feat: add thing"         "missing space after colon"
assert_normalize "fix(scope):bug fix"      "fix(scope): bug fix"     "missing space with scope"
assert_normalize "feat!:breaking"          "feat!: breaking"         "missing space with bang"

# Combined normalizations
assert_normalize "Feature:new thing"       "feat: new thing"         "alias + uppercase + missing space"
assert_normalize "BUGFIX(core):oops"       "fix(core): oops"        "all-caps alias + scope + missing space"
assert_normalize "Doc!:breaking docs"      "docs!: breaking docs"   "doc alias + bang + missing space"

# ---------- Validation tests ----------

# Valid messages
assert_valid "feat: add new feature"                 "basic feat"
assert_valid "fix(scope): correct bug"               "fix with scope"
assert_valid "docs: update readme"                   "docs type"
assert_valid "chore(deps): update dependencies"      "chore with scope"
assert_valid "feat!: breaking change"                "breaking change"
assert_valid "fix(my scope): with spaces in scope"   "scope with spaces"
assert_valid "refactor(a/b): path scope"             "scope with slash"
assert_valid "ci: update workflow"                   "ci type"
assert_valid "test: add unit tests"                  "test type"
assert_valid "perf: improve speed"                   "perf type"
assert_valid "style: format code"                    "style type"
assert_valid "build: update config"                  "build type"
assert_valid "revert: undo change"                   "revert type"

# Invalid messages
assert_invalid "add new feature"                     "no type prefix"
assert_invalid "feat:"                               "empty description"
assert_invalid "feat: "                              "space-only after colon (no desc)"
assert_invalid "unknown: some change"                "unknown type"
assert_invalid "feat add thing"                      "missing colon"
assert_invalid "$(printf 'feat: %0.s.' {1..70})"    "over 72 chars"

# ---------- Exemption tests ----------

assert_exempt "Merge branch 'feature' into main"   "merge commit"
assert_exempt "Merge pull request #42"              "merge PR"
assert_exempt "fixup! feat: original"               "fixup prefix"
assert_exempt "squash! feat: original"              "squash prefix"
assert_exempt "bd sync something"                   "bd sync"

assert_not_exempt "feat: normal commit"             "normal commit"
assert_not_exempt "fix: not exempt"                 "fix commit"
assert_not_exempt "Merging things"                  "Merging (not Merge )"

# ---------- End-to-end: normalize then validate ----------

e2e_pass() {
  local input="$1"
  local label="$2"
  local normalized
  normalized=$(normalize "$input")
  if validate "$normalized"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    ERRORS="${ERRORS}FAIL [e2e] ${label}\n  input:      '${input}'\n  normalized: '${normalized}'\n  expected valid after normalization\n\n"
  fi
}

e2e_pass "Feat: add thing"              "e2e uppercase type"
e2e_pass "feature: new thing"           "e2e alias expansion"
e2e_pass "BUGFIX(core):fix crash"       "e2e combined transforms"
e2e_pass "Doc!:update api docs"         "e2e doc alias + bang + space"

# ---------- Report ----------

printf '\n--- Results ---\n'
printf 'Passed: %d\n' "$PASS"
printf 'Failed: %d\n' "$FAIL"

if [ "$FAIL" -gt 0 ]; then
  printf '\n--- Failures ---\n'
  printf '%b' "$ERRORS"
  exit 1
fi

printf 'All tests passed.\n'
