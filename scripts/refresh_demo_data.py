"""Rebuild the demo snapshot used by the public website.

Run it from the project folder:

    .venv/bin/python scripts/refresh_demo_data.py

What it does: reads data/demo/demo_portfolio.json, downloads fresh prices from Yahoo
Finance (it needs the internet and ignores the cache), and rewrites data/demo/demo_prices.csv
and data/demo/demo_meta.json. Then commit those two files so the website picks them up.

This script is only a thin wrapper. The real work lives in signaldesk/data.py
(``refresh_demo_snapshot``), where it can be tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import signaldesk` work when this file is run as a script from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signaldesk import data  # noqa: E402  (must come after the sys.path line above)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the demo price snapshot from Yahoo Finance.")
    parser.add_argument(
        "--demo-dir",
        default=None,
        help="folder holding demo_portfolio.json (default: data/demo in the project)",
    )
    args = parser.parse_args(argv)
    demo_dir = Path(args.demo_dir) if args.demo_dir else data._default_demo_dir()

    try:
        meta = data.refresh_demo_snapshot(demo_dir)
        demo = data.load_demo(demo_dir)  # read it back: proves the files we wrote are valid
    except (ValueError, OSError) as error:
        # ValueError covers PriceDataError and ProviderUnavailableError (nothing came back);
        # OSError covers a folder or file we cannot write (read-only disk, no permission).
        print(f"Could not refresh the demo data: {error}", file=sys.stderr)
        return 1

    gaps = data.find_gaps(demo.prices)
    missing_days = data.find_missing_days(demo.prices)

    print(f"Wrote {demo_dir / data.DEMO_PRICES_FILE}")
    print(f"Wrote {demo_dir / data.DEMO_META_FILE}")
    print(f"  portfolio : {demo.name}")
    print(f"  tickers   : {', '.join(meta['tickers'])}")
    print(f"  rows      : {meta['rows']}")
    print(f"  dates     : {meta['start']} to {meta['as_of']}")
    print(f"  source    : {meta['source']}")
    print(f"  created   : {meta['created_at']}")
    print()
    print("Missing prices per ticker (find_gaps):")
    print(gaps.to_string())
    print()
    print(f"Weekdays absent from the table (find_missing_days): {len(missing_days)}")
    print("  (most are normal US market holidays; look for anything that is not)")
    for day in missing_days:
        print(f"  {day:%Y-%m-%d} {day:%a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
