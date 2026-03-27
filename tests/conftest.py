import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gen-ref",
        action="store_true",
        default=False,
        help="Generate reference outputs instead of comparing against them.",
    )


@pytest.fixture(scope="session")
def gen_ref(request):
    return request.config.getoption("--gen-ref")
