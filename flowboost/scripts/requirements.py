import subprocess
from pathlib import Path


def generate():
    req_p = Path("requirements")
    if not req_p.exists():
        req_p.mkdir(exist_ok=True)

    reqs = "poetry export -f requirements.txt --output requirements/requirements.txt --without dev --without-hashes"
    subprocess.run([reqs], shell=True, check=True)

    reqs_dev = "poetry export -f requirements.txt --output requirements/requirements_dev.txt --with dev --without-hashes"
    subprocess.run([reqs_dev], shell=True, check=True)
