#!/bin/bash

# Phase 1: Knowledge Graph Construction
echo "Starting Agentic-Graph-RAG Pipeline..."

# Ensure we are executing from the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Phase 1
python3 -m phase1.data_loader