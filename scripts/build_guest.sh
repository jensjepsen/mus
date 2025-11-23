rm -r wasm_deps
mkdir wasm_deps
uv pip compile pyproject.toml --output-file wasm_deps/requirements.txt
uvx --native-tls pip install --target wasm_deps --platform any --platform wasi_0_0_0_wasm32 --python-version "3.12" --only-binary :all: --index-url https://benbrandt.github.io/wasi-wheels/ --extra-index-url https://pypi.org/simple --upgrade -r wasm_deps/requirements.txt

echo "Building guest wasm module"
cp -r src/mus wasm_deps/mus

echo "Componentizing guest"
uvx componentize-py -d ./src/mus/guest -w muswasm componentize guest.main -o src/mus/guest.wasm --stub-wasi -p wasm_deps -p src/mus
echo "Done componentizing guest"

uvx --from wasmtime==34.0.0 python -m wasmtime.bindgen src/mus/guest.wasm --out-dir src/mus/guest/bindings