#!/usr/bin/env python3
"""Command-line prediction for one JSON record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import ARTIFACT_DIR, RAW_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON object with all 21 raw features")
    parser.add_argument(
        "--model", type=Path, default=ARTIFACT_DIR / "diabetes_screening_pipeline.joblib"
    )
    parser.add_argument(
        "--metadata", type=Path, default=ARTIFACT_DIR / "model_metadata.json"
    )
    args = parser.parse_args()
    record = json.loads(args.input.read_text())
    missing = sorted(set(RAW_FEATURES) - set(record))
    extra = sorted(set(record) - set(RAW_FEATURES))
    if missing or extra:
        raise SystemExit(f"Schema mismatch. Missing={missing}; extra={extra}")
    model = joblib.load(args.model)
    metadata = json.loads(args.metadata.read_text())
    score = float(model.predict_proba(pd.DataFrame([record], columns=RAW_FEATURES))[0, 1])
    threshold = float(metadata["operating_threshold"])
    print(
        json.dumps(
            {
                "risk_score": round(score, 6),
                "operating_threshold": threshold,
                "screen_positive": score >= threshold,
                "model": metadata["selected_model"],
                "disclaimer": "Not a diagnosis or calibrated future-risk probability; clinician review is required.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

