#!/bin/bash

# This script is used to build the guest wasm module

curl -Ls https://raw.githubusercontent.com/extism/python-pdk/main/install.sh | bash

rm -r wasm_deps
mkdir wasm_deps
uv pip compile pyproject.toml -o wasm_deps/temp-reqs.txt
uvx --native-tls pip install --upgrade --force --target wasm_deps --platform any --platform wasi_0_0_0_wasm32 --python-version "3.12" --only-binary :all: --index-url https://benbrandt.github.io/wasi-wheels/ --extra-index-url https://pypi.org/simple pydantic-core
uvx --native-tls pip install -r wasm_deps/temp-reqs.txt
ls wasm_deps

echo "Building guest wasm module"
cp -r src/mus wasm_deps/mus
ls wasm_deps
PYTHONPATH=wasm_deps extism-py src/mus/guest/main.py -o guest.wasm