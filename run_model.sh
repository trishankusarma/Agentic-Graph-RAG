#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

exec >"$LOG_FILE" 2>&1

echo "Starting Agentic-Graph-RAG Pipeline..."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# python3 -m kg.graph_store
# python -m experiments.validate_repair --no-use_semantic
# python -m experiments.validate_repair --use_semantic
python3 -m experiments.evaluate_em_f1