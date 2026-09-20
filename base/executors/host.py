"""Device-independent host process helpers."""
import json
import subprocess
from typing import Any


def docker_inspect(image: str) -> dict[str, Any]:
    proc = subprocess.run(["docker", "image", "inspect", image], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"image is unavailable: {image}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)[0]
