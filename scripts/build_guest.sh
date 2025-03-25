#!/bin/bash

# This script is used to build the guest wasm module

poetry install --target wasm_deps
PYTHONPATH=./wasm_deps extism-py src