#!/bin/bash

# This script is used to build the guest wasm module

pip install --target wasm_deps -e .
echo "Building guest wasm module"

PYTHONPATH=./wasm_deps extism-py src/mus/guest/main.py -o guest.wasm