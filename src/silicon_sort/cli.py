"""Command line entry points."""

from __future__ import annotations

import argparse
import sys

from silicon_sort.errors import Cancelled, SiliconSortError
from silicon_sort.pipeline import Pipeline, resolve_settings
from silicon_sort.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon-sort", description="Spike sorting for brainstem silicon probes")
    sub = parser.add_subparsers(dest="command", required=True)

    gui = sub.add_parser("gui", help="Open the desktop app")
    gui.add_argument("--dat", default="", help="Optional recording to preload")
    gui.add_argument("--cfg", default="", help="Optional geometry file to preload")

    run = sub.add_parser("run", help="Run the pipeline from a settings JSON file")
    run.add_argument("settings", help="Path to pipeline_settings.json")
    run.add_argument("--output", default="", help="Phy output directory")

    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from silicon_sort.gui.app import launch

            return launch(args.dat, args.cfg)
        if args.command == "run":
            return _run(args.settings, args.output)
    except Cancelled:
        print("Cancelled.", file=sys.stderr)
        return 2
    except (SiliconSortError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    parser.error(f"Unknown command {args.command}")
    return 2


def _run(settings_path: str, output: str) -> int:
    settings = resolve_settings(load_settings(settings_path))
    if output:
        settings.output_dir = output
    pipeline = Pipeline(settings)

    def progress(fraction: float, message: str) -> None:
        print(f"{fraction * 100:5.1f}%  {message}", file=sys.stderr)

    try:
        target = pipeline.run_all(progress=progress)
    finally:
        pipeline.close()
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
