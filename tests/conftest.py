from pytest import fixture


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store"
    )
    parser.addoption(
        "--dist",
        action="store"
    )

