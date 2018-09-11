#!/bin/bash
for filename in model_configs/*.json; do
    python "/home/lawson/Workspace/AutoDef/scripts/build-model.py" "-f" "True" "-c" "$filename" "-o" "./trained_models_avg_error/$(basename "$filename" .json)"
done