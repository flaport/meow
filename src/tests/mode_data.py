import json
import os

MODE_DATA = json.load(
    open(os.path.join(os.path.dirname(__file__), "assets", "model.json"))
)

if __name__ == "__main__":
    print(MODE_DATA["neff"])
