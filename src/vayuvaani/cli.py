"""Command-line interface.

    vayu demo                       # synthetic data, end to end (no API keys)
    vayu fetch --start 2023-01-01 --end 2026-10-31
    vayu build
    vayu train
    vayu forecast                   # live 48-hour forecast
    vayu forecast --replay 2026-10-20T08:00
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import forecast, pipeline, train
from .config import load_settings


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def cmd_fetch(args, s):
    pipeline.fetch(s, args.start, args.end, skip_fires=args.no_fires)


def cmd_build(args, s):
    pipeline.build(s)


def cmd_train(args, s):
    report = train.run(s, pipeline.load_processed(s))
    print("\n" + train.format_report(report))


def cmd_forecast(args, s):
    out = forecast.replay(s, args.replay) if args.replay is not None else forecast.live(s)
    print("\nWorst hour in the next 24 hours (median forecast, 10–90% range):\n")
    print(forecast.format_forecast(out))
    print(f"\nFull forecast: {s.path('reports') / 'latest_forecast.csv'}")


def cmd_demo(args, s):
    end = pd.Timestamp(args.end)
    start = end - pd.Timedelta(days=args.days)
    print("Generating SYNTHETIC data (made-up numbers, for testing only)...")
    pipeline.fetch_demo(s, start.date().isoformat(), end.date().isoformat())
    pipeline.build(s)
    report = train.run(s, pipeline.load_processed(s))
    print("\n" + train.format_report(report))
    out = forecast.replay(s)
    print("\nReplay forecast, worst hour in the next 24 hours:\n")
    print(forecast.format_forecast(out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vayu", description="VayuVaani air quality forecasting")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--config", help="path to settings.json")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download real data into data/raw")
    f.add_argument("--start", required=True, help="YYYY-MM-DD")
    f.add_argument("--end", required=True, help="YYYY-MM-DD")
    f.add_argument("--no-fires", action="store_true", help="skip NASA FIRMS (no key needed)")
    f.set_defaults(func=cmd_fetch)

    sub.add_parser("build", help="clean and merge raw data").set_defaults(func=cmd_build)
    sub.add_parser("train", help="train and evaluate the model").set_defaults(func=cmd_train)

    fc = sub.add_parser("forecast", help="48-hour forecast for every station")
    fc.add_argument("--replay", nargs="?", const="", default=None,
                    help="forecast from a past hour using observed weather (YYYY-MM-DDTHH:MM)")
    fc.set_defaults(func=cmd_forecast)

    d = sub.add_parser("demo", help="run everything on synthetic data")
    d.add_argument("--days", type=int, default=400)
    d.add_argument("--end", default="2026-10-31")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    settings = load_settings(args.config)
    try:
        args.func(args, settings)
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
