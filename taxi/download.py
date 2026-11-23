"""Download the NYC TLC yellow taxi trip records listed in params.yaml.

Source: NYC Taxi & Limousine Commission (TLC) Trip Record Data,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Raw files are never committed: this script plus `dvc repro` rebuilds them.
"""

from __future__ import annotations

import requests

from taxi.common import RAW_DIR, params

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for month in params()["data"]["months"]:
        dest = RAW_DIR / f"yellow_tripdata_{month}.parquet"
        if dest.exists():
            print(f"skip {dest.name} (already here)")
            continue
        url = URL.format(month=month)
        print(f"download {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                f.writelines(r.iter_content(1 << 20))
            tmp.rename(dest)


if __name__ == "__main__":
    main()
