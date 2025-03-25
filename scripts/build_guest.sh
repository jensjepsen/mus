#!/bin/bash

# This script is used to build the guest wasm module

cp -r src/mus wasm_deps/mus
PYTHONPATH=./wasm_deps extism-py src/mus/guest.py -o guest.wasm