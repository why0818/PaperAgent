from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_packages() -> None:
    """Add project-local pip target to sys.path when it exists."""
    project_root = Path(__file__).resolve().parents[1]
    package_dir = project_root / ".packages"
    if package_dir.exists():
        package_path = str(package_dir)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

