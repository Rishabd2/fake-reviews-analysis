"""Record the row order the original bag-of-words notebook used, as Ott file ids.

The saved notebook read a CSV export of the corpus whose order came from
os.listdir on the original machine. A random 80/20 split depends on that order,
so src/reproduce_notebook.py needs it to repeat the 90.3% run exactly.
This script matches each CSV row to its op_spam_v1.4 file and writes only
file ids (no review text) to data/notebook_row_order.txt.

    python scripts/derive_notebook_order.py /path/to/Chicago_Hotel_Reviews.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from corpus import load_ott  # noqa: E402


def main(csv_path: str) -> None:
    csv = pd.read_csv(csv_path)
    ott = load_ott()
    by_text: dict = {}
    for t, lab, src in zip(ott["text"], ott["label"], ott["source"]):
        by_text.setdefault((t.strip(), lab), []).append(src)
    order = []
    for text, label in zip(csv["Ori_Review"], csv["Label"]):
        key = (str(text).strip(), "human_fake" if label == 1 else "truthful")
        # a few reviews appear twice in the corpus; assign their files in sorted order
        order.append(by_text[key].pop(0))
    assert len(set(order)) == 1600
    (REPO / "data" / "notebook_row_order.txt").write_text("\n".join(order) + "\n")
    print("wrote", len(order), "ids")


if __name__ == "__main__":
    main(sys.argv[1])
