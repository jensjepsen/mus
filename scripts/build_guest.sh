#!/bin/bash

# This script is used to build the guest wasm module

python -m pip install --target wasm_deps -e .
python -m pip install --target wasm_deps --platform any --platform wasi_0_0_0_wasm32 --python-version "3.12" --only-binary :all: --index-url https://benbrandt.github.io/wasi-wheels/ --extra-index-url https://pypi.org/simple --upgrade pydantic-core

echo "Building guest wasm module"
cp -r src/mus wasm_deps/mus
ls wasm_deps
PYTHONPATH=./wasm_deps extism-py src/mus/guest/main.py -o guest.wasm