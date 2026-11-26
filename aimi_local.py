"""Start AIMi with its locally installed dependencies."""
from pathlib import Path
import site
import sys

packages = str(Path(__file__).resolve().parent / ".aimi-packages")
if sys.prefix == sys.base_prefix:
    site.addsitedir(packages)
    if packages in sys.path:
        sys.path.remove(packages)
        sys.path.insert(0, packages)

from utils.inputs import aimbot

if __name__ == "__main__":
    aimbot(ENABLE_AIMBOT=True)
