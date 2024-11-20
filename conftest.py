"""
Setup sybil to run tests in markdown files.
This conftest.py needs to be in the root, to find all the markdown files.
"""

from sybil import Sybil
import pytest
from unittest import mock
from src.mus.llm.mock_client import MockLLM
from sybil.parsers.markdown import PythonCodeBlockParser


@pytest.fixture(autouse=True)
def mock_clients():
    import sys
    import src.mus
    m = mock.Mock(wraps=src.mus)
    m.AnthropicLLM = MockLLM
    sys.modules["mus"] = m
    yield m


pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(future_imports=['print_function']),
    ],
    pattern='*.md',
    fixtures=['mock_clients'],
).pytest()
