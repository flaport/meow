import json
from pathlib import Path

MODE_DATA = json.loads(
    (Path(__file__).resolve().parent / "assets" / "model.json").read_text()
)
