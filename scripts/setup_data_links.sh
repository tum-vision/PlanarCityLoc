#!/bin/bash
# Link externally obtained benchmark data and checkpoints without overwriting files.
# Usage:
#   bash scripts/setup_data_links.sh /path/to/data /path/to/checkpoints
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/setup_data_links.sh /path/to/data /path/to/checkpoints" >&2
  exit 1
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$(readlink -f "$1")"
CHECKPOINT_ROOT="$(readlink -f "$2")"
[[ -d "$DATA_ROOT" ]] || { echo "Missing data directory: $DATA_ROOT" >&2; exit 1; }
[[ -d "$CHECKPOINT_ROOT" ]] || { echo "Missing checkpoint directory: $CHECKPOINT_ROOT" >&2; exit 1; }

declare -a link_sources=() link_targets=()

plan_link() {
  local source_path="$1" target_path="$2"
  if [[ -e "$target_path" || -L "$target_path" ]]; then
    if [[ "$(readlink -f "$target_path")" != "$(readlink -f "$source_path")" ]]; then
      echo "Existing path is preserved: $target_path. Use its current source or move it explicitly." >&2
      return 1
    fi
  else
    link_sources+=("$source_path")
    link_targets+=("$target_path")
  fi
}

# Check every required benchmark file before creating any links. Files already
# present in this checkout are accepted only when their recorded hash matches.
while read -r expected relative; do
  [[ -n "$expected" ]] || continue
  [[ "$relative" == data/* && "$relative" != *..* ]] || {
    echo "Invalid benchmark path in data/SHA256SUMS" >&2; exit 1;
  }
  target_path="$REPO_ROOT/$relative"
  source_path="$DATA_ROOT/${relative#data/}"
  candidate="$source_path"
  if [[ -e "$target_path" || -L "$target_path" ]]; then
    candidate="$target_path"
  fi
  [[ -f "$candidate" ]] || { echo "Missing benchmark file: $candidate" >&2; exit 1; }
  actual="$(sha256sum -- "$candidate")"
  [[ "${actual%% *}" == "$expected" ]] || {
    echo "Benchmark identity mismatch: $relative. Check the dataset package and map version." >&2; exit 1;
  }
  if [[ ! -e "$target_path" && ! -L "$target_path" ]]; then
    plan_link "$source_path" "$target_path"
  fi
done < "$REPO_ROOT/data/SHA256SUMS"

plan_queries() {
  local relative="$1"
  local source_dir="$DATA_ROOT/$relative/Query_image"
  if [[ ! -d "$source_dir" ]]; then
    source_dir="$DATA_ROOT/$relative/Query_image_full"
  fi
  if [[ ! -d "$source_dir" ]]; then
    echo "Missing Query_image or Query_image_full under $DATA_ROOT/$relative" >&2
    return 1
  fi
  plan_link "$source_dir" "$REPO_ROOT/data/$relative/Query_image"
}

plan_queries UAVD4L-LoD/inTraj
plan_queries UAVD4L-LoD/outTraj
plan_queries Swiss-EPFL/inPlace
plan_queries Swiss-EPFL/outPlace
plan_link "$CHECKPOINT_ROOT" "$REPO_ROOT/checkpoints"

for i in "${!link_sources[@]}"; do
  mkdir -p "$(dirname "${link_targets[$i]}")"
  ln -s -- "${link_sources[$i]}" "${link_targets[$i]}"
  echo "Linked ${link_targets[$i]}"
done
echo "Benchmark files verified. Data and checkpoints are ready."
