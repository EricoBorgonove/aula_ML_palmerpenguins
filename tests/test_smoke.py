def test_imports():
    """Smoke test to ensure the test runner can import minimal packages."""
    import pandas as pd
    import sklearn

    assert pd.__version__
    assert sklearn.__version__
