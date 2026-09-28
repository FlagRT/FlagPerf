"""Load pure, repository-owned PR0 helpers without processing site hooks."""
import importlib.util
from pathlib import Path

PROFILE = Path(__file__).parent / "xpytorch_2.9_p800_candidate"


def load(name):
    spec = importlib.util.spec_from_file_location("flagperf_pr0_" + name, PROFILE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapping = load("device_mapping")
runtime = load("verify_runtime")
