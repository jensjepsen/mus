#!/bin/bash

# This script is used to build the guest wasm module

curl -Ls https://raw.githubusercontent.com/extism/python-pdk/main/install.sh | bash

python -m pip install --platform any --platform wasi_0_0_0_wasm32 --python-version "3.12" --only-binary :all: --target wasm_deps https://github.com/benbrandt/wasi-wheels/releases/download/pydantic-core%2Fv2.33.0/pydantic_core-2.33.0-cp312-cp312-wasi_0_0_0_wasm32.whl

echo "Building guest wasm module"
cp -r src/mus wasm_deps/mus
ls wasm_deps
PYTHONPATH=wasm_deps extism-py test.py -o guest.wasm