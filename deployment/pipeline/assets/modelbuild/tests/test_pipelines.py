import pytest


# @pytest.mark.xfail
@pytest.mark.skip(reason="no way of currently testing this")
def test_that_you_wrote_tests():
    assert False, "No tests written"

@pytest.mark.skip(reason="no way of currently testing this")
def test_pipelines_importable():
    import pipelines  # noqa: F401
