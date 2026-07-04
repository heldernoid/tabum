"""Select ~500 classification + ~500 regression datasets from OpenML for the
Phase 5 benchmark, writing validation/benchmark_suite.json.

Selection is deliberately opinionated:
- dedupe by name (keep lowest data id = original version)
- drop auto-generated bloat families (BNG(...) has ~1000s of synthetic clones;
  QSAR-TID-* are near-identical drug-target sets — capped, not banned;
  fri_c* / autoUniv are synthetic generators)
- shape sanity: 60..500k rows, 2..500 features, classes 2..100 (model table
  limit), minority class >= 10 rows (must appear in a 2k-row train split)
- score real-world-ness: moderate size, class balance, name heuristics;
  take top-N by score
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

EXCLUDE_NAME = re.compile(r"^(BNG\(|fri_c|autoUniv|RandomRBF|LED\(|SEA\(|Stagger|"
                          r"Agrawal|Hyperplane|AirlinesCodrnaAdult|Waveform)", re.I)
QSAR = re.compile(r"^QSAR-TID-", re.I)
QSAR_CAP = 20
OUR_IDS = {3, 6, 11, 12, 15, 23, 29, 31, 37, 38, 44, 50, 54, 151, 182, 188, 307,
           469, 1462, 1464, 1489, 1494, 6332, 23381, 1590, 1461, 189, 209, 546,
           541, 507, 183, 22, 28, 46, 32, 1510, 227, 40701}  # tag, don't exclude


def fetch_all() -> list[dict]:
    out, offset = [], 0
    while True:
        url = (f"https://www.openml.org/api/v1/json/data/list/status/active"
               f"/limit/10000/offset/{offset}")
        with urllib.request.urlopen(url, timeout=120) as r:
            batch = json.load(r)["data"]["dataset"]
        out.extend(batch)
        if len(batch) < 10000:
            return out
        offset += 10000


def qualities(d: dict) -> dict:
    return {q["name"]: float(q["value"]) for q in d.get("quality", [])
            if q.get("value") not in (None, "")}


def score(d: dict, q: dict, kind: str) -> float:
    n, f = q["NumberOfInstances"], q["NumberOfFeatures"]
    s = 0.0
    s += 2.0 if 500 <= n <= 100_000 else (1.0 if 100 <= n else 0.0)
    s += 1.0 if 3 <= f <= 300 else 0.0
    miss = q.get("NumberOfMissingValues", 0.0)
    s += 0.5 if miss == 0 else 0.25  # fully observed slightly preferred
    if kind == "cls":
        mino, majo = q.get("MinorityClassSize", 0), q.get("MajorityClassSize", n)
        if majo > 0:
            s += 1.0 * min(1.0, (mino / majo) * 4)  # reward non-degenerate balance
    name = d["name"]
    if re.search(r"\d{4,}$", name) or QSAR.match(name):
        s -= 0.5  # bulk-family suffix
    s -= 0.25 * name.count("_")  # generated names tend to be underscore-heavy
    return s


def main():
    raw = fetch_all()
    print(f"active datasets: {len(raw)}")
    seen_names, qsar_count = set(), 0
    cls_rows, reg_rows = [], []
    for d in sorted(raw, key=lambda x: int(x["did"])):  # lowest did = original
        name = d["name"]
        if name.lower() in seen_names or EXCLUDE_NAME.match(name):
            continue
        if d.get("format", "").lower() != "arff":
            continue
        q = qualities(d)
        if not {"NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"} <= q.keys():
            continue
        n, f, c = q["NumberOfInstances"], q["NumberOfFeatures"], q["NumberOfClasses"]
        if not (60 <= n <= 500_000 and 2 <= f <= 500):
            continue
        if QSAR.match(name):
            qsar_count += 1
            if qsar_count > QSAR_CAP:
                continue
        entry = {"did": int(d["did"]), "name": name, "n_rows": int(n),
                 "n_features": int(f), "n_classes": int(c),
                 "in_tuning_set": int(d["did"]) in OUR_IDS}
        if 2 <= c <= 100 and q.get("MinorityClassSize", 0) >= 10:
            entry["score"] = score(d, q, "cls")
            cls_rows.append(entry)
            seen_names.add(name.lower())
        elif c == 0:
            entry["score"] = score(d, q, "reg")
            reg_rows.append(entry)
            seen_names.add(name.lower())

    cls_rows.sort(key=lambda e: -e["score"])
    reg_rows.sort(key=lambda e: -e["score"])
    suite = {"classification": cls_rows[:500], "regression": reg_rows[:500]}
    out = Path("validation/benchmark_suite.json")
    out.write_text(json.dumps(suite, indent=1))
    for kind, rows in suite.items():
        cmax = max((e["n_classes"] for e in rows), default=0)
        print(f"{kind}: {len(rows)} selected (candidates: "
              f"{len(cls_rows) if kind == 'classification' else len(reg_rows)}), "
              f"max classes {cmax}")
    dist = {}
    for e in suite["classification"]:
        b = ("2" if e["n_classes"] == 2 else "3-10" if e["n_classes"] <= 10
             else "11-26" if e["n_classes"] <= 26 else "27-100")
        dist[b] = dist.get(b, 0) + 1
    print("class-count distribution:", dict(sorted(dist.items())))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
