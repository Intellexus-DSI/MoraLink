from importlib.metadata import version

import moralink


def test_package_version_is_public_release_version():
    assert moralink.__version__ == "1.0.0"
    assert version("moralink") == "1.0.0"
