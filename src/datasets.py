"""Dataset loaders for the eight study tasks.

Triangular fuzzy outputs always use (center, left_spread, right_spread).
No preprocessing is fitted here; fold-local scaling belongs to evaluation.py.
"""
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data"
PATHS = {
    1: DATA_ROOT / "native" / "example1.xlsx",
    2: DATA_ROOT / "native" / "example2.xlsx",
    3: DATA_ROOT / "native" / "example3.xlsx",
    4: DATA_ROOT / "native" / "example4.xlsx",
    5: DATA_ROOT / "native" / "example5.xlsx",
}
NAMES = {
    1: "Example 1 (30-pair time series)",
    2: "Example 2 (30 fuzzy-input pairs)",
    3: "Example 3 (8-pair illustration)",
    4: "Example 4 (184-pair benchmark)",
    5: "Example 5 (25-pair fuzzy-input benchmark)",
}


def _xlsx(path, sheet=None):
    """Read the small legacy xlsx files without an Excel-engine dependency."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Follow data/data_sources.md"
        )
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in item.findall(".//m:t", ns))
                for item in root.findall("m:si", ns)
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        mapping = {entry.attrib["Id"]: entry.attrib["Target"] for entry in rels}
        sheets = workbook.findall(".//m:sheet", ns)
        selected = next(
            (entry for entry in sheets if entry.attrib.get("name") == sheet), sheets[0]
        )
        relation = selected.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = mapping[relation].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        worksheet = ET.fromstring(archive.read(target))
        rows = []
        for row in worksheet.findall(".//m:sheetData/m:row", ns):
            values, last = [], 0
            for cell in row.findall("m:c", ns):
                letters = re.match(r"[A-Z]+", cell.attrib.get("r", "A1")).group()
                column = 0
                for character in letters:
                    column = column * 26 + ord(character) - 64
                values += [""] * max(0, column - last - 1)
                last = column
                value_node = cell.find("m:v", ns)
                value = "" if value_node is None else (value_node.text or "")
                if cell.attrib.get("t") == "s" and value.isdigit():
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                values.append(value)
            if any(str(value).strip() for value in values):
                rows.append(values)
        return rows


def _records(path, sheet=None):
    rows = _xlsx(path, sheet)
    return [{str(key): value for key, value in zip(rows[0], row)} for row in rows[1:]]


def _tfn(value):
    parsed = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
    return parsed[:3] if len(parsed) >= 3 else [parsed[0], parsed[1], parsed[1]]


def load_example(number):
    """Load native Examples 1--5 in the representation used in the study."""
    rows = _records(PATHS[number], "Numeric_Data" if number == 5 else None)
    if number == 1:
        features = np.array([[float(row["xi"])] for row in rows])
        targets = np.array([_tfn(row["yi"]) for row in rows])
    elif number == 2:
        raw = np.array([
            [float(row[f"X_i{i}_{part}"]) for i in (1, 2) for part in ("core", "left", "right")]
            for row in rows
        ])
        features = np.array([
            [values[0], values[1] + values[2], values[3], values[4] + values[5], values[0] * values[3]]
            for values in raw
        ])
        targets = np.array([
            [float(row[f"Y_i_{part}"]) for part in ("core", "left", "right")]
            for row in rows
        ])
    elif number == 3:
        features = np.array([[float(row["x_core"])] for row in rows])
        targets = np.array([
            [float(row["y_core"]), float(row["y_spread"]), float(row["y_spread"])]
            for row in rows
        ])
    elif number == 4:
        features = np.array([[float(row["X"])] for row in rows])
        targets = np.array([_tfn(row["y"]) for row in rows])
    elif number == 5:
        raw = np.array([
            [float(row[f"x{i}_{part}"]) for i in range(1, 10) for part in ("center", "left", "right")]
            for row in rows
        ])
        features = np.hstack((raw[:, ::3], (raw[:, 1::3] + raw[:, 2::3]) / 2))
        targets = np.array([
            [float(row[f"y_{part}"]) for part in ("center", "left", "right")]
            for row in rows
        ])
    else:
        raise ValueError(f"Unknown native example: {number}")
    if features.ndim != 2 or targets.shape != (len(features), 3) or np.any(targets[:, 1:] < 0):
        raise ValueError(f"Invalid Example {number} representation")
    return features.astype(float), targets.astype(float)


def fuzzify_crisp_target(target):
    """Deterministic synthetic-TFN rule: (y, .08 max(|y|,1), .12 max(|y|,1))."""
    target = np.asarray(target, dtype=float)
    scale = np.maximum(np.abs(target), 1.0)
    return np.column_stack((target, 0.08 * scale, 0.12 * scale))


def load_public_dataset(name):
    """Load the three synthetically fuzzified public tasks."""
    from sklearn.datasets import load_diabetes
    import pandas as pd

    if name == "Diabetes":
        features, target = load_diabetes(return_X_y=True)
    elif name == "EnergyEfficiency":
        table = pd.read_excel(DATA_ROOT / "public" / "ENB2012_data.xlsx")
        features = table.iloc[:, :8].to_numpy(float)
        target = table["Y1"].to_numpy(float)  # heating load, not cooling load
    elif name == "ConcreteStrength":
        table = pd.read_csv(DATA_ROOT / "public" / "Concrete_Data.csv")
        features = table.iloc[:, :8].to_numpy(float)
        target = table.iloc[:, 8].to_numpy(float)
    else:
        raise ValueError(f"Unknown public dataset: {name}")
    original_n = len(features)
    selected = np.arange(original_n)
    metadata = {
        "original_n": original_n,
        "used_n": len(features),
        "selected_indices": selected.tolist(),
        "status": "SYNTHETICALLY_FUZZIFIED_CRISP_DATA",
    }
    return np.asarray(features, float), fuzzify_crisp_target(target), metadata
