from unittest import mock
from src.mus.llm.mock_client import MockLLM
orig_import = __import__
def mock_clients():
    def m(name, *args):
        breakpoint()
        return orig_import(name, *args)
    with mock.patch("__builtin__.__import__", side_effect=m):
        yield