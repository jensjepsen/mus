## v0.26.6 (2025-12-25)

### Fix

- tool-results-in-google
- don't serialize metadata and pass to llm
- don't serialize metadata and pass to llm
- don't serialize metadata and pass to llm

## v0.26.5 (2025-12-11)

### Fix

- openai streaming tool inputs

## v0.26.4 (2025-12-09)

### Fix

- Merge pull request #51 from jensjepsen/fix/tool-not-found-error
- fallback function as function
- fallback function as function

## v0.26.3 (2025-12-09)

### Fix

- Merge pull request #50 from jensjepsen/fix/tool-not-found-error
- toolnotfounderror
- toolnotfounderror
- toolnotfounderror

## v0.26.2 (2025-12-08)

### Fix

- handle thought signatures from google

## v0.26.1 (2025-12-08)

### Fix

- Merge pull request #49 from jensjepsen/fix/google-streaming-tool-inputs
- simluate streaming tool input updates in google genai until supp…
- simluate streaming tool input updates in google genai until supported by provider

## v0.26.0 (2025-12-03)

### Feat

- async bedrock
- async bedrock
- async bedrock

### Refactor

- remove duplication

## v0.25.3 (2025-11-30)

### Fix

- openai-no-stream
- don't pass stream_options to openai when streaming
- don't include stream options for openai when not streaming

## v0.25.2 (2025-11-28)

### Fix

- delta tool input

## v0.25.1 (2025-11-28)

### Fix

- Merge pull request #45 from jensjepsen/fix/empty-json-function-input
- empty content input when parsing json in bedrock tool calls
- empty content input when parsing json in bedrock tool calls

## v0.25.0 (2025-11-27)

### Feat

- streaming function calls
- streaming function calls

## v0.24.4 (2025-11-23)

### Fix

- pin wasmtime version in build script
- allow returning ToolValue from tools

## v0.24.3 (2025-11-01)

### Fix

- use toolvalue in llm clients

## v0.24.2 (2025-11-01)

### Fix

- don't coerce tool results to json
- don't coerce tool results to json
- don't coerce tool results to json
- don't coerce tool results to json

## v0.24.1 (2025-10-06)

### Fix

- tool values

## v0.24.0 (2025-10-03)

### Feat

- allow structured return from sandbox

## v0.23.1 (2025-09-30)

### Fix

- poll instead of creating new loop
- poll instead of creating new loop

## v0.23.0 (2025-09-28)

### Feat

- sandbox inputs
- sandbox inputs
- sandbox inputs
- sandbox inputs
- refactor sandbox for speed

## v0.22.10 (2025-09-28)

### Fix

- sandbox context
- sandbox context
- sandbox context

## v0.22.9 (2025-09-28)

### Fix

- don't destructure dc completely

## v0.22.8 (2025-09-28)

### Fix

- unstructure and structure sandbox function args properly

## v0.22.7 (2025-09-27)

### Fix

- throw errors in host
- throw errors in host
- remove string with metadata - should be a dedicated return type instead

## v0.22.6 (2025-09-27)

### Fix

- handle sequence types

## v0.22.5 (2025-09-27)

### Fix

- avoid unnecessary forward refs

## v0.22.4 (2025-09-27)

### Fix

- handle mapping collections and any
- handle mapping collections and any

## v0.22.3 (2025-09-27)

### Fix

- use get_type_hints instead of annotations, remove unneccesary forward refs

## v0.22.2 (2025-09-27)

### Fix

- handle pydantic models when generating schemas

## v0.22.1 (2025-09-27)

### Fix

- new version of anthropic sdk contained breaking changes - use omit instead of notgiven
- reexport delta types from __init__

## v0.22.0 (2025-09-25)

### Feat

- remove-json-pickle
- remove jsonpickle dep
- remove jsonpickle dep
- remove jsonpickle dep
- remove jsonpickle dep

## v0.21.0 (2025-09-22)

### Feat

- string with metadata return type for tools
- string with metadata
- string with metadata
- string with metadata

### Fix

- bump openai

## v0.20.1 (2025-09-18)

### Fix

- support for literal in schema
- support literal in schema
- support literal in schema

## v0.20.0 (2025-09-17)

### Feat

- multiple models and functions for sandbox
- multiple models and functions for sandbox
- external functions
- external functions
- multiple models in sandbox

### Fix

- ruff
- sandbot in readme
- sandbot in readme

## v0.19.8 (2025-09-08)

### Fix

- check if choices exist on openai responses

## v0.19.7 (2025-09-08)

### Fix

- include usage in openai responses

## v0.19.6 (2025-09-08)

### Fix

- update to use max_completion_tokens for openai

## v0.19.5 (2025-09-07)

### Fix

- openai tools
- openai tools
- openai tools
- openai tools

## v0.19.4 (2025-09-07)

### Fix

- openai
- openai
- mistral
- mistral
- openai
- openai client

## v0.19.3 (2025-07-09)

### Fix

- verifying subscripted generics

## v0.19.2 (2025-07-08)

### Fix

- verify annotated types
- annotated types in verification
- annotated types in verification

## v0.19.1 (2025-07-02)

### Fix

- relax cattrs

## v0.19.0 (2025-07-01)

### Feat

- mistral
- mistral
- mistral
- mistral
- mistral

## v0.18.1 (2025-06-25)

### Fix

- Merge pull request #24 from jensjepsen/dependabot/pip/wasmtime-34.0.0

## v0.18.0 (2025-06-25)

### Feat

- function and fill input validation

### Fix

- lint

### Refactor

- minor cleanup

## v0.17.0 (2025-06-24)

### Feat

- fill and function validation
- verify function args
- verify function args
- verify function args
- verify fill

### Refactor

- serialize using cattrs in state

## v0.16.7 (2025-06-19)

### Fix

- fillable type hints

## v0.16.6 (2025-06-19)

### Fix

- re-export llm from main module

## v0.16.5 (2025-06-19)

### Refactor

- test_llm -> test_bot

## v0.16.4 (2025-06-19)

### Fix

- MockLLM -> StubLLM

## v0.16.3 (2025-06-19)

### Fix

- re-export mockllm from mus

## v0.16.2 (2025-06-19)

### Fix

- LLMClient -> LLM

## v0.16.1 (2025-06-19)

### Fix

- llm -> bot

### Refactor

- llm to bot
- llm to bot

## v0.16.0 (2025-06-18)

### Feat

- google gemini

## v0.15.0 (2025-06-18)

### Feat

- prompt and tool caching
- caching

### Fix

- deps
- cache written name
- remove unused import

### Refactor

- cleanup test

## v0.14.5 (2025-06-10)

### Fix

- catch wrong args to tool

## v0.14.4 (2025-06-10)

### Fix

- release

## v0.14.3 (2025-06-10)

### Fix

- release

## v0.14.2 (2025-06-10)

### Fix

- release

## v0.14.1 (2025-06-10)

### Fix

- jensjepsen/dependabot/pip/wasmtime-33.0.0

## v0.14.0 (2025-06-10)

### Feat

- mcp
- mcp
- mcp

### Refactor

- cleanup types
- parse tools
- don't depend on pydantic
- don't depend on pydantic
- don't depend on pydantic
- use local schema
- simplify passing tools to clients

## v0.13.0 (2025-05-19)

### Feat

- wasmtime and fuel
- wasmtime and fuel
- wasmtime and fuel

### Fix

- pyright config
- pyright config
- don't typecheck bindigns
- remove extism

## v0.12.2 (2025-05-01)

### Fix

- merge and prune deltas

## v0.12.1 (2025-05-01)

### Fix

- don't include empty text

## v0.12.0 (2025-04-17)

### Feat

- fill with prefill
- prefill fill
- prefill echo

### Fix

- types

## v0.11.0 (2025-04-15)

### Feat

- simplify llm and state classes
- simplify llm and state
- move state and llm from mus instance

### Refactor

- statemanager call to init

## v0.10.0 (2025-04-06)

### Feat

- dynamic system prompt
- dynamic system prompt
- dynamic system prompt

### Fix

- tests
- merge

## v0.9.4 (2025-04-06)

### Fix

- simplify models and clients
- simplify clients and models
- simplify clients and models

## v0.9.3 (2025-04-05)

### Fix

- don't import sandbox functionality when in a sandbox
- move sandbox to __init__.py and update readme with sandboxing

## v0.9.2 (2025-04-05)

### Fix

- remove stubs

## v0.9.1 (2025-04-05)

### Fix

- dedent

## v0.9.0 (2025-04-04)

### Feat

- sandbox decorator

### Fix

- proper indent of sandboxed code
- proper indent of sandboxed code
- proper indent of sandboxed code
- include print in sandbox

## v0.8.0 (2025-04-01)

### Feat

- wasm sandbox
- sandbox
- sandbox
- sandbox
- sandobx
- sandbox
- sandbox
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- extism
- sandbox
- sandbox

### Fix

- update build script
- add missing build script
- use uv in pr
- update pydantic version
- sandbox
- build
- merge main
- install pydantic core

## v0.7.8 (2025-03-31)

### Fix

- Merge pull request #4 from jensjepsen/dependabot/pip/jsonpickle-gte-3.4.2-and-lt-5.0.0
- Merge pull request #5 from jensjepsen/dependabot/pip/typer-gte-0.13.0-and-lt-0.16.0

## v0.7.7 (2025-03-30)

### Fix

- add pydantic

## v0.7.6 (2025-03-28)

### Fix

- use setuptools

## v0.7.5 (2025-03-28)

### Fix

- use uv
- merge main
- merge main
- use uv

## v0.7.4 (2025-03-24)

### Fix

- quotes in f string

## v0.7.3 (2025-03-14)

### Fix

- use threading for async in boto bedrock

## v0.7.2 (2025-03-11)

### Fix

- bump
- bump
- bump

## v0.7.1 (2025-03-10)

### Fix

- cleaup types
- cleanup types

## v0.7.0 (2025-03-09)

### Feat

- add reasoning support to bedrock

## v0.6.5 (2025-03-05)

### Fix

- dependencies

## v0.6.4 (2025-03-05)

### Fix

- optional import of extra deps

## v0.6.3 (2025-03-05)

### Fix

- don't update optional deps

## v0.6.2 (2025-03-05)

### Fix

- optional import of extra deps

## v0.6.1 (2025-03-05)

### Fix

- add all extra

## v0.6.0 (2025-03-04)

### Feat

- optional dependencies
- optional dependencies
- optional dependencies
- optional dependencies

### Fix

- update lockfile
- update lockfile
- update lockfile
- update lockfile

## v0.5.2 (2025-03-01)

### Fix

- bump versions
- move sybil to dev deps

## v0.5.1 (2025-03-01)

### Fix

- move sybil to dev deps

## v0.5.0 (2025-02-14)

### Feat

- add openai

## v0.4.0 (2025-02-07)

### Feat

- add stop_sequences

## v0.3.9 (2025-01-17)

### Fix

- don't trigger main on bump

## v0.3.8 (2025-01-17)

### Fix

- don't trigger main on bump
- don't trigger main on bump

## v0.3.7 (2025-01-17)

### Fix

- don't trigger main on bump

## v0.3.6 (2025-01-17)

### Fix

- don't trigger main on bump

## v0.3.5 (2025-01-17)

### Fix

- update lockfile

## v0.3.4 (2025-01-16)

### Fix

- try release pipeline

## v0.3.3 (2025-01-16)

### Fix

- try release pipeline
- try release pipeline
- try release pipeline
- try release pipeline

## v0.3.2 (2025-01-16)

### Fix

- try release pipeline

## v0.3.1 (2025-01-16)

### Fix

- try release pipeline

## v0.3.0 (2025-01-16)

### Feat

- test release pipeline

## v0.2.0 (2025-01-16)

### Feat

- prefilling assistant answer
- don't stream for non interactive responses
- add bedrock client
- add bedrock client
- add bedrock client
- all async
- everything async
- pyodide patching
- add usage to deltas
- more args for queries
- reorder params
- more streaming params

### Fix

- broken .fill
- add missing test
- deref schemas
- deref schemas
- don't use tool choice for bedrock
- fun decorator should not be async
- fun decorator should not be async
- bump anthropic
- default values for kw args in client
- add missing action
- test action
- broken tests

### Refactor

- actions
- actions
- actions
- actions
- split actions
- split actions
- more types
- use internal messages and reconstruct external
