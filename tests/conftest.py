"""Use fresh workspace-local scratch space across Windows account contexts."""
from pathlib import Path
import tempfile


def pytest_configure(config):
    if config.option.basetemp is None:
        # A unique directory avoids removing another account's prior pytest run.
        config.option.basetemp = tempfile.mkdtemp(
            prefix='.test_tmp_', dir=Path(__file__).resolve().parents[1])
