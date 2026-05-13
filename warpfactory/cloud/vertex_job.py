"""Backward-compatible Vertex AI entrypoint.

Use ``python -m warpfactory.cloud.run_job --execution-target vertex`` for new
workflows. Vertex autopackaging can still point at this module.
"""

from warpfactory.cloud.run_job import build_parser, main, run

__all__ = ["build_parser", "main", "run"]


if __name__ == "__main__":
    main()
