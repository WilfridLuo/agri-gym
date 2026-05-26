#!/usr/bin/env python3
"""AgriGym entry point."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from run import main  # noqa: E402


if __name__ == "__main__":
    main()
