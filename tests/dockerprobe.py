"""Is there a daemon, and is the sandbox image built?

The probe lived in test_heart.py, guarding TestSandboxLive. It was never
applied to the sandbox tests in test_sandbox.py, test_local_slots.py or
TestSandboxWrap, so those have been failing in CI since 15 September --
GitHub's runner has a docker daemon but no heart-agent:latest, which is
exactly the case the probe was written to detect.

The image is checked, never pulled: a test suite that silently downloads
gigabytes is a test suite people stop running. Build it with
`docker build -t heart-agent:latest .` and these un-skip.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def docker_usable() -> bool:
    if not shutil.which("docker"):
        return False
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from heart.sandbox import DEFAULT_IMAGE

    image = os.environ.get("HEART_SANDBOX_IMAGE", DEFAULT_IMAGE)
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0


DOCKER_USABLE = docker_usable()
