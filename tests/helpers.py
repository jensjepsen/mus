from unittest import mock
from mus.llm.mock_client import MockLLM
orig_import = __import__
def mock_clients():
    def m(name, *args):
        return orig_import(name, *args)
    with mock.patch("__builtin__.__import__", side_effect=m):
        yield