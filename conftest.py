"""
Setup sybil to run tests in markdown files.
This conftest.py needs to be in the root, to find all the markdown files.
"""

from sybil import Sybil
import pytest
from unittest import mock
from mus.llm.mock_client import StubLLM
from sybil.parsers.markdown import PythonCodeBlockParser


@pytest.fixture(autouse=True)
def mock_clients():
    import sys
    import mus
    import mus.dbos

    original = sys.modules["mus"]
    m = mock.Mock(wraps=mus)
    m.AnthropicLLM = StubLLM
    # A Mock is not a package, so `import mus.dbos` cannot resolve while it is
    # standing in. Expose the real submodule so docs can still reach it via
    # `from mus import dbos`.
    m.dbos = sys.modules["mus.dbos"]
    sys.modules["mus"] = m
    yield m
    sys.modules["mus"] = original


pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(future_imports=['print_function']),
    ],
    pattern='*.md',
    fixtures=['mock_clients'],
).pytest()
