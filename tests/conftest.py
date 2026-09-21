"""Keep test scratch files isolated from shared or restricted Windows temp dirs."""
from uuid import uuid4


def pytest_configure(config):
    if config.option.basetemp is None:
        parent = config.rootpath / "tmp" / "pytest"
        parent.mkdir(parents=True, exist_ok=True)
        # Every run gets a new directory; never clear another run's files.
        config.option.basetemp = str(parent / uuid4().hex)
