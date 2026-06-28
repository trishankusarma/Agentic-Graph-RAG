#!/bin/bash

# Phase 1: Knowledge Graph Construction
echo "Starting Agentic-Graph-RAG Pipeline..."

# Ensure we are executing from the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Phase 1
# python3 -m kg.data_loader
python3 -m kg.hypergraph_builder