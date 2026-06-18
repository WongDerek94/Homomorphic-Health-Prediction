#!/usr/bin/env python3
"""Regenerate preprocessed train/test CSVs from raw Kaggle splits.

Transforms raw ``Training.csv`` / ``Testing.csv`` into the committed
``Training_preprocessed.csv`` / ``Testing_preprocessed.csv`` files used by
``dev.py``, ``app.py``, and ``scripts/generate_evidence.py``.

Usage (from the repository root):

    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --verify-only

The script verifies byte-identical output against golden SHA256 digests in
``data/testset_checksum.txt``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

RAW_TRAINING = DATA_DIR / "Training.csv"
RAW_TESTING = DATA_DIR / "Testing.csv"
OUT_TRAINING = DATA_DIR / "Training_preprocessed.csv"
OUT_TESTING = DATA_DIR / "Testing_preprocessed.csv"

GOLDEN_CHECKSUMS = {
    OUT_TRAINING.name: "99619ffcb7f422eacbf40b34d093d98f43c4913ef8e5b447c9783967ee52820b",
    OUT_TESTING.name: "d40a49ce3f1de2e09ca1ecc9657d6a841b72f1a312dae0ccb540616b9991c248",
}

# 128 feature columns in the exact order required by the deployed FHE model.
FEATURE_COLUMNS = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "excess_body_fat",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremeties",
    "excessive_hunger",
    "frequent_unprotected_sexual_intercourse_with_multiple_partners",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_(typhus)",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "abnormal_menstruation",
    "dischromic_patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "stomach_bleeding",
    "distention_of_abdomen",
    "chronic_alcohol_abuse",
    "severe_fluid_overload",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurving",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]

OUTPUT_HEADER = FEATURE_COLUMNS + ["prognosis", "prognosis_encoded"]

# Raw column name -> preprocessed feature column (for renamed headers).
RAW_RENAMES = {
    "spotting_ urination": "spotting_urination",
    "diarrhoea": "diarrhea",
    "obesity": "excess_body_fat",
    "extra_marital_contacts": "frequent_unprotected_sexual_intercourse_with_multiple_partners",
    "foul_smell_of urine": "foul_smell_of_urine",
    "dischromic _patches": "dischromic_patches",
    "toxic_look_(typhos)": "toxic_look_(typhus)",
    "history_of_alcohol_consumption": "chronic_alcohol_abuse",
    "scurring": "scurving",
}

# Explicit prognosis label overrides (all others use str.title()).
PROGNOSIS_OVERRIDES = {
    "(vertigo) Paroymsal  Positional Vertigo": "Paroxymsal Positional Vertigo",
    "Dimorphic hemmorhoids(piles)": "Dimorphic Hemmorhoids (Piles)",
    "Peptic ulcer diseae": "Peptic Ulcer",
}


def build_feature_index_map(raw_header: list[str]) -> dict[str, int]:
    """Map each preprocessed feature column to its raw column index."""
    fluid_idxs = [i for i, name in enumerate(raw_header) if name == "fluid_overload"]
    index_map: dict[str, int] = {}

    for feature in FEATURE_COLUMNS:
        if feature == "severe_fluid_overload":
            index_map[feature] = fluid_idxs[1]
            continue

        for raw_name, prep_name in RAW_RENAMES.items():
            if prep_name == feature:
                index_map[feature] = raw_header.index(raw_name)
                break
        else:
            index_map[feature] = raw_header.index(feature)

    return index_map


def feature_value(raw_row: list[str], feature: str, raw_header: list[str], index_map: dict[str, int]) -> int:
    """Return the binary feature value, applying dropped-column merge rules."""
    value = int(float(raw_row[index_map[feature]]))

    if feature == "anxiety":
        depression_idx = raw_header.index("depression")
        value = 1 if value or int(float(raw_row[depression_idx])) else 0
    elif feature == "stomach_pain":
        belly_idx = raw_header.index("belly_pain")
        value = 1 if value or int(float(raw_row[belly_idx])) else 0

    return value


def prognosis_index(raw_header: list[str]) -> int:
    return raw_header.index("prognosis")


def normalize_prognosis(raw_label: str) -> str:
    stripped = raw_label.strip()
    if stripped in PROGNOSIS_OVERRIDES:
        normalized = PROGNOSIS_OVERRIDES[stripped]
    else:
        normalized = stripped.title()
    # Preserve trailing whitespace present in the raw Kaggle labels (e.g. "Diabetes ").
    trailing = raw_label[len(raw_label.rstrip()) :]
    return normalized + trailing


def format_feature(value: str) -> str:
    return f"{int(float(value)):.1f}"


def build_encoding_map(normalized_labels: list[str]) -> dict[str, int]:
    unique = sorted(set(normalized_labels), key=str.lower)
    return {label: code for code, label in enumerate(unique)}


def read_raw_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header, rows


def preprocess_rows(
    raw_header: list[str],
    raw_rows: list[list[str]],
    encoding_map: dict[str, int],
) -> list[list[str]]:
    feature_map = build_feature_index_map(raw_header)
    prog_idx = prognosis_index(raw_header)
    output_rows: list[list[str]] = []

    for raw_row in raw_rows:
        features = [
            format_feature(feature_value(raw_row, col, raw_header, feature_map))
            for col in FEATURE_COLUMNS
        ]
        prognosis = normalize_prognosis(raw_row[prog_idx])
        encoded = str(encoding_map[prognosis])
        output_rows.append(features + [prognosis, encoded])

    return output_rows


def write_preprocessed(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_HEADER)
        writer.writerows(rows)


def sha256_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_file(path: Path, expected_rows: int | None = None) -> bool:
    digest = sha256_digest(path)
    expected = GOLDEN_CHECKSUMS[path.name]
    with path.open(newline="") as handle:
        row_count = sum(1 for _ in csv.reader(handle)) - 1
    col_count = len(OUTPUT_HEADER)

    name = "train" if "Training" in path.name else "test"
    print(f"{name}: file={path.name} rows={row_count} cols={col_count} sha256={digest}")

    ok = digest == expected
    if not ok:
        print(f"  FAIL: expected sha256={expected}", file=sys.stderr)
    if expected_rows is not None and row_count != expected_rows:
        print(f"  FAIL: expected rows={expected_rows}, got {row_count}", file=sys.stderr)
        ok = False
    return ok


def run_preprocess() -> bool:
    train_header, train_rows = read_raw_rows(RAW_TRAINING)
    test_header, test_rows = read_raw_rows(RAW_TESTING)

    train_labels = [normalize_prognosis(row[prognosis_index(train_header)]) for row in train_rows]
    encoding_map = build_encoding_map(train_labels)

    train_out = preprocess_rows(train_header, train_rows, encoding_map)
    test_out = preprocess_rows(test_header, test_rows, encoding_map)

    write_preprocessed(OUT_TRAINING, train_out)
    write_preprocessed(OUT_TESTING, test_out)

    ok_train = verify_file(OUT_TRAINING, expected_rows=4920)
    ok_test = verify_file(OUT_TESTING, expected_rows=42)
    return ok_train and ok_test


def run_verify_only() -> bool:
    missing = [p for p in (OUT_TRAINING, OUT_TESTING) if not p.exists()]
    if missing:
        for path in missing:
            print(f"Missing file: {path}", file=sys.stderr)
        return False

    ok_train = verify_file(OUT_TRAINING, expected_rows=4920)
    ok_test = verify_file(OUT_TESTING, expected_rows=42)
    return ok_train and ok_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw disease-prediction CSVs.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing preprocessed files against golden SHA256 digests.",
    )
    args = parser.parse_args()

    success = run_verify_only() if args.verify_only else run_preprocess()
    if success:
        print("\nPreprocessing verification passed.")
        return 0

    print("\nPreprocessing verification failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
