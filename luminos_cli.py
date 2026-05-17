#!/usr/bin/env python3
"""
Luminos CLI — convert a film negative scan to a positive TIFF.

Usage:
    python luminos_cli.py input.tif output.tif
    python luminos_cli.py input.NEF output.tif --exposure 0.5 --wb 1.1 1.0 0.85
"""

import argparse
from luminos.core.pipeline import convert_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Luminos: film negative converter")
    parser.add_argument("input", help="Input file (RAW or TIFF)")
    parser.add_argument("output", help="Output TIFF path")
    parser.add_argument(
        "--exposure", type=float, default=0.0,
        metavar="STOPS",
        help="Exposure adjustment in EV stops (default: 0.0)",
    )
    parser.add_argument(
        "--wb", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("R", "G", "B"),
        help="White balance RGB multipliers (default: 1.0 1.0 1.0)",
    )
    args = parser.parse_args()

    convert_file(
        args.input,
        args.output,
        exposure_stops=args.exposure,
        white_balance=tuple(args.wb),
    )


if __name__ == "__main__":
    main()
