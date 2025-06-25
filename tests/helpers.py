from unittest import mock
orig_import = __import__
def mock_clients():
    def m(name, *args):
        return orig_import(name, *args)
    with mock.patch("__builtin__.__import__", side_effect=m):
        yield