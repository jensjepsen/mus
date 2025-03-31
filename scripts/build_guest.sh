rm -r wasm_deps
mkdir wasm_deps
uv pip compile pyproject.toml --output-file wasm_deps/requirements.txt
uvx --native-tls pip install --target wasm_deps --platform any --platform wasi_0_0_0_wasm32 --python-version "3.12" --only-binary :all: --index-url https://benbrandt.github.io/wasi-wheels/ --extra-index-url https://pypi.org/simple --upgrade -r wasm_deps/requirements.txt

echo "Building guest wasm module"
cp -r src/mus wasm_deps/mus
ls wasm_deps
PYTHONPATH=./wasm_deps extism-py src/mus/guest/main.py -o src/mus/guest.wasm