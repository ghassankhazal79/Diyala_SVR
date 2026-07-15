#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""استخدام نموذج SVR المحفوظ للتنبؤ من ملف CSV جديد."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="التنبؤ بالتبخر من ملف CSV جديد")
    parser.add_argument("--model", default="outputs/models/svr_evaporation_model.joblib")
    parser.add_argument("--input", required=True, help="CSV يحتوي station, month والمتغيرات المناخية")
    parser.add_argument("--output", default="new_predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(Path(args.model))
    model = bundle["model"]
    features = bundle["features"]

    data = pd.read_csv(args.input)
    required = ["station", "month", *[f for f in features if f not in {"month_sin", "month_cos"}]]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"الأعمدة المطلوبة غير موجودة: {missing}")

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["predicted_evaporation"] = model.predict(data[[*features, "station"]])
    data.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"تم حفظ التنبؤات في: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
