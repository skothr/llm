#!/usr/bin/env bash
# theory/series/build.sh — clean-rebuild all 5 papers and collect their
# rendered PDFs into theory/series/dist/<N>-<topic>.pdf as symlinks.
#
# Usage:
#   bash theory/series/build.sh             # clean + build all 5 + collect
#   bash theory/series/build.sh collect     # collect only (skip build)
#
# Sequential build is required: each paper's main.tex declares
# \externaldocument cross-refs to its four siblings via xr-hyper, so
# their main.aux files must exist before a paper's later passes can
# resolve cross-paper labels. We do two full sweeps over all five
# papers so cross-refs settle (3 pdflatex passes per paper + bibtex).
#
# pdflatex's exit code is unreliable under -interaction=nonstopmode
# (it exits non-zero on benign hyperref/xr-hyper interaction warnings
# while still emitting a clean PDF), so we chain with `;` not `&&`.

set -u

SERIES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="${SERIES_DIR}/dist"

# slug per paper, used as the file name in dist/
declare -A SLUG=(
  [1]="architecture"
  [2]="training"
  [3]="reasoning"
  [4]="interpretability"
  [5]="evaluation-alignment"
)

build_one() {
  local n="$1"
  local dir="${SERIES_DIR}/paper-${n}"
  echo "  paper-${n} ..."
  (
    cd "${dir}"
    rm -f main.aux main.log main.out main.bbl main.blg main.toc
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    bibtex main > /dev/null 2>&1
  )
}

resolve_one() {
  # 3 extra pdflatex passes to settle cross-paper labels via xr-hyper.
  local n="$1"
  local dir="${SERIES_DIR}/paper-${n}"
  (
    cd "${dir}"
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
  )
}

collect() {
  mkdir -p "${DIST}"
  # Remove any stale symlinks first so renaming a slug doesn't leave
  # dangling old-named entries in dist/.
  find "${DIST}" -maxdepth 1 -type l -name '*.pdf' -delete 2>/dev/null
  local fail=0
  for n in 1 2 3 4 5; do
    local src="${SERIES_DIR}/paper-${n}/main.pdf"
    local dst="${DIST}/${n}-${SLUG[$n]}.pdf"
    if [[ ! -f "${src}" ]]; then
      echo "  MISSING: ${src} (paper-${n} did not produce a PDF)"
      fail=1
      continue
    fi
    # Relative symlink so dist/ is portable across moves of the repo root.
    ln -sf "../paper-${n}/main.pdf" "${dst}"
    local pages
    pages=$(pdfinfo "${src}" 2>/dev/null | awk '/^Pages:/ {print $2}')
    echo "  ${n}-${SLUG[$n]}.pdf -> paper-${n}/main.pdf  (${pages}pp)"
  done
  return "${fail}"
}

case "${1:-all}" in
  collect)
    echo "Collecting PDFs into dist/ ..."
    collect
    exit "$?"
    ;;
  all|"")
    echo "Sweep 1: clean + pdflatex + bibtex per paper ..."
    for n in 1 2 3 4 5; do build_one "${n}"; done
    echo "Sweep 2: 3 pdflatex passes per paper to settle xr-hyper cross-refs ..."
    for n in 1 2 3 4 5; do resolve_one "${n}"; done
    # Some papers (paper-2 historically) need one more pass when the
    # glossary section length crosses a pagination boundary. Cheap to
    # rerun; idempotent at convergence.
    echo "Sweep 3: extra pass for any papers still resolving ..."
    for n in 1 2 3 4 5; do
      p="${SERIES_DIR}/paper-${n}/main.pdf"
      unresolved=$(pdftotext "${p}" - 2>/dev/null | grep -c '(?,')
      if [[ "${unresolved}" -gt 0 ]]; then
        echo "  paper-${n}: ${unresolved} unresolved cites — extra pass"
        (cd "${SERIES_DIR}/paper-${n}" && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1)
      fi
    done
    echo "Collect:"
    collect
    echo ""
    echo "Final state:"
    for n in 1 2 3 4 5; do
      p="${SERIES_DIR}/paper-${n}/main.pdf"
      pages=$(pdfinfo "${p}" 2>/dev/null | awk '/^Pages:/ {print $2}')
      unresolved=$(pdftotext "${p}" - 2>/dev/null | grep -c '(?,')
      echo "  paper-${n}: ${pages}pp unresolved-cites=${unresolved}"
    done
    ;;
  *)
    echo "usage: bash $(basename "$0") [all|collect]" >&2
    exit 2
    ;;
esac
