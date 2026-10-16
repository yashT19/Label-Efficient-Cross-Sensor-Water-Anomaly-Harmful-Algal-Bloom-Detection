"""Compatibility launcher for the config-driven PACE OCI pipeline."""

from __future__ import annotations

import sys

from run import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--site", "lake_erie", "--sensor", "pace"])
    main()
