#!/bin/bash

# This script is used to build the guest wasm module

pip install --target wasm_deps -e .
PYTHONPATH=./wasm_deps extism-py src/mus/guest.py -o guest.wasm