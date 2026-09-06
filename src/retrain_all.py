"""Walk-forward retraining: refreshes all production models + stamps versions.

Run weekly (cron Sunday 06Z):  python src/retrain_all.py
  - weather: both cities (fast, ~1 min each)
  - crypto direction: BTC + ETH (~5-8 min each; GRU/stack excluded — no value)
Writes trained_models/versions.json {model_key: UTC-timestamp} which traders
stamp onto every paper trade (model_version column) for per-version W/L.
"""
import json
from datetime import datetime, timezone

from signals import MODEL_DIR


from tracker import stamp


def main():
    import train_wx
    import train_direction
    for city in ("NYC", "CHI"):
        train_wx.main(city)
        print(f"stamped wx_{city}={stamp(f'wx_{city}')}", flush=True)
    for asset in ("BTC", "ETH"):
        train_direction.main(asset)
        print(f"stamped dir_{asset}={stamp(f'dir_{asset}')}", flush=True)
    print("versions:", json.dumps(json.loads((MODEL_DIR / 'versions.json').read_text()), indent=1))


if __name__ == "__main__":
    main()
