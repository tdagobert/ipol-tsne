#!/usr/bin/env bash
  
set -eu

binfolder=$1

### fetch parameters
# Always provide the working directory
params=""
# Then add remaining parameters
# Start from the second argument: $1 is the bin folder
for p in "${@:1}"; do
    params="$params $p"
done

python3 $binfolder/tsne-demo.py $params


