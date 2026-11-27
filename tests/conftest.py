import sys
from pathlib import Path

# Add ltx-core source to sys.path so tests can import without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LTX_CORE_SRC = _REPO_ROOT / "packages" / "ltx-core" / "src"
if str(_LTX_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(_LTX_CORE_SRC))
