#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

exec >"$LOG_FILE" 2>&1

echo "Starting Agentic-Graph-RAG Pipeline..."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 -m kg.graph_store