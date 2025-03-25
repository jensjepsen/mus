#!/bin/bash

# This script is used to build the guest wasm module

pip install -e --target wasm_deps .
PYTHONPATH=./wasm_deps extism-py src/mus/guest.py -o guest.wasm