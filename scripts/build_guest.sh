#!/bin/bash

# This script is used to build the guest wasm module

curl -Ls https://raw.githubusercontent.com/extism/python-pdk/main/install.sh | bash

python -m pip install --target wasm_deps --platform any --platform wasi_0_0_0_wasm32 --python-version "3.12" --only-binary :all: --index-url https://benbrandt.github.io/wasi-wheels/ --extra-index-url https://pypi.org/simple pydantic-core
python -m pip install --target wasm_deps -e .


echo "Building guest wasm module"
cp -r src/mus wasm_deps/mus
ls wasm_deps
PYTHONPATH=./wasm_deps extism-py src/mus/guest/main.py -o guest.wasm