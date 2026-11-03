import unittest


def assert_raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def skip(msg):
    raise unittest.SkipTest(msg)  # honoured by pytest as a skip
