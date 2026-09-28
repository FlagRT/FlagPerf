"""Device-independent host process helpers."""
import json
import subprocess
from typing import Any
from executors.lifecycle import HostCommands


def docker_inspect(image: str, *, commands=None) -> dict[str, Any]:
    result = (commands or HostCommands()).checked(["docker", "image", "inspect", image], privileged=True)
    records = json.loads(result['stdout'])
    if len(records) != 1:
        raise RuntimeError("image inspection must describe exactly one image")
    return records[0]
