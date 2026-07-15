#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diyala Monthly Evaporation Prediction
=====================================
برنامج متكامل لاستخراج البيانات المناخية من ملف «موحد.xlsx»، وتنظيفها،
ثم تدريب نموذج Support Vector Regression (SVR) ومقارنته مع نماذج أخرى.

المخرجات الرئيسة:
- قاعدة بيانات شهرية نظيفة.
- مقارنة SVR مع Seasonal Climatology, Persistence, Linear Regression,
  Random Forest, Gradient Boosting.
- تحليل حسب المحطة والشهر.
- اختبار Wilcoxon للفروق في الأخطاء.
- Ablation Study لمجموعات المتغيرات.
- Leave-One-Station-Out spatial validation.
- جداول Excel/CSV ورسوم PNG ونموذج محفوظ بصيغة joblib.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings(
    "ignore",
    message="Sparkline Group extension is not supported and will be removed",
)

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

FEATURE_LABELS_AR = {
    "tmean": "درجة الحرارة الاعتيادية",
    "tmin": "درجة الحرارة الصغرى",
    "tmax": "درجة الحرارة العظمى",
    "rh": "الرطوبة النسبية",
    "wind": "سرعة الرياح",
    "rainfall": "الأمطار",
    "sunshine": "السطوع الشمسي",
    "solar_rad": "الإشعاع الشمسي",
    "month_sin": "جيب الشهر",
    "month_cos": "جيب تمام الشهر",
}

# الكلمات التي يعتمد عليها البرنامج لاكتشاف كل جدول داخل أوراق Excel.
SECTION_ALIASES: dict[str, list[str]] = {
    "tmean": [
        "درجة الحرارة الاعتيادية", "الحرارة الاعتيادية",
        "MEAN AIR TEMP", "MEAN AIR TEMPERATURE",
    ],
    "tmin": [
        "درجة الحرارة الصغرى", "الحرارة الصغرى",
        "MEAN MIN TEMP", "MEAN MIN. TEMP",
    ],
    "tmax": [
        "درجة الحرارة العظمى", "الحرارة العظمى",
        "MEAN MAX TEMP", "MEAN MAX.",
    ],
    "rh": [
        "الرطوبة النسبية", "الرطوبة خانقين",
        "MEAN RELATIVE HUMIDITY", "RELATIVE HUMIDITY",
    ],
    "wind": ["سرعة الرياح", "سرعة رياح", "WIND SPEED"],
    "rainfall": [
        "MONTHLY RAINFALL TOTALS", "ELEMENT: RAINFALL", "ELEMENT RAINFALL",
    ],
    "sunshine": ["السطوع الشمسي", "SUNSHINE"],
    "solar_rad": ["الاشعاع الشمسي", "الإشعاع الشمسي", "SOLAR RADIATION"],
    "evaporation": ["MONTHLY EVAPORATION TOTALS", "حوض التبخر"],
    "eto": ["التبخر نتح", "EVAPOTRANSPIRATION"],
}

BASE_CLIMATE_FEATURES = [
    "tmean", "tmin", "tmax", "rh", "wind", "rainfall", "sunshine", "solar_rad"
]


@dataclass
class ExtractedSection:
    feature: str
    marker_row: int
    header_row: int
    records: list[tuple[int, list[float]]]
    title: str


@dataclass
class RunConfig:
    excel_path: Path
    output_dir: Path
    start_year: int = 1994
    end_year: int = 2016
    test_years: int = 5
    target: str = "evaporation"
    random_state: int = 42
    min_feature_coverage: float = 0.65
    remove_target_outliers: bool = False
    skip_tuning: bool = False
    quick: bool = False


def normalize_text(value: Any) -> str:
    """توحيد النص العربي/الإنجليزي لاكتشاف عناوين الأقسام."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .upper()
    )
    return re.sub(r"[\s\-_/().:%°˚ْ]+", "", text)


def clean_number(value: Any) -> float:
    """تحويل القيم النصية مثل 24,4 أو 5..0 إلى قيمة رقمية آمنة."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text or text.upper() in {"M", "NA", "N/A", "NULL", "NONE", "-", "—"}:
        return np.nan

    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = text.replace("،", ",").replace("٫", ".").replace(",", ".")
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _is_year_header(value: Any) -> bool:
    text = normalize_text(value)
    return text in {"YEAR", "YM", "Y/M", "السنه", "سنه"}


def discover_sections(raw: pd.DataFrame) -> dict[str, ExtractedSection]:
    """اكتشاف جداول المتغيرات داخل ورقة واحدة آليًا."""
    normalized_aliases = {
        key: [normalize_text(alias) for alias in aliases]
        for key, aliases in SECTION_ALIASES.items()
    }
    found: dict[str, ExtractedSection] = {}
    rows, cols = raw.shape

    for feature, aliases in normalized_aliases.items():
        candidates: list[ExtractedSection] = []
        for row_idx in range(rows):
            row_text = " ".join(
                normalize_text(v) for v in raw.iloc[row_idx, : min(cols, 40)].tolist()
            )
            if not row_text or not any(alias and alias in row_text for alias in aliases):
                continue

            # لا نخلط جدول التبخر-نتح مع التبخر الحوضي.
            if feature == "evaporation" and (
                normalize_text("التبخر نتح") in row_text
                or normalize_text("EVAPOTRANSPIRATION") in row_text
            ):
                continue

            for header_idx in range(row_idx, min(row_idx + 12, rows)):
                first_value = raw.iat[header_idx, 0] if cols else None
                if not _is_year_header(first_value):
                    continue

                records: list[tuple[int, list[float]]] = []
                data_idx = header_idx + 1
                while data_idx < rows:
                    year_value = clean_number(raw.iat[data_idx, 0])
                    if np.isnan(year_value) or not (1900 <= year_value <= 2100):
                        break

                    monthly_values = [
                        clean_number(raw.iat[data_idx, col]) if col < cols else np.nan
                        for col in range(1, 13)
                    ]
                    if np.isfinite(monthly_values).sum() >= 6:
                        records.append((int(year_value), monthly_values))
                    data_idx += 1

                if records:
                    candidates.append(
                        ExtractedSection(
                            feature=feature,
                            marker_row=row_idx,
                            header_row=header_idx,
                            records=records,
                            title=row_text,
                        )
                    )
                break

        if candidates:
            # اختيار القسم ذي أكبر عدد من السنوات، ثم الأسبق في الورقة.
            candidates.sort(key=lambda section: (-len(section.records), section.marker_row))
            found[feature] = candidates[0]

    return found


def section_to_long(station: str, section: ExtractedSection) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, values in section.records:
        for month, value in enumerate(values, start=1):
            rows.append(
                {
                    "station": station,
                    "year": int(year),
                    "month": int(month),
                    section.feature: value,
                }
            )
    return pd.DataFrame(rows)


def extract_workbook(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """قراءة جميع أوراق الملف ودمج الجداول في قاعدة شهرية واحدة."""
    workbook = pd.ExcelFile(excel_path)
    all_station_frames: list[pd.DataFrame] = []
    discovery_rows: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        sections = discover_sections(raw)
        if not sections:
            warnings_list.append(f"لم يتم اكتشاف أي جدول مناخي في الورقة: {sheet_name}")
            continue

        # فحص تعارض اسم بلدروز/بدرة الموجود في الورقة الأصلية.
        normalized_sheet_text = " ".join(
            normalize_text(v)
            for v in raw.iloc[:, : min(raw.shape[1], 40)].astype(object).to_numpy().ravel()
            if pd.notna(v)
        )
        if normalize_text("BADRA") in normalized_sheet_text and normalize_text("بلدروز") in normalize_text(sheet_name):
            warnings_list.append(
                "الورقة المسماة «بلدروز» تتضمن العبارة STATION: BADRA؛ "
                "يجب التحقق من هوية المحطة قبل النشر النهائي. البرنامج يعتمد اسم الورقة «بلدروز»."
            )

        merged: pd.DataFrame | None = None
        for feature, section in sections.items():
            long_df = section_to_long(sheet_name, section)
            merged = long_df if merged is None else merged.merge(
                long_df, on=["station", "year", "month"], how="outer"
            )
            discovery_rows.append(
                {
                    "station": sheet_name,
                    "feature": feature,
                    "marker_row_excel": section.marker_row + 1,
                    "header_row_excel": section.header_row + 1,
                    "first_year": section.records[0][0],
                    "last_year": section.records[-1][0],
                    "years_count": len(section.records),
                    "title_detected": section.title,
                }
            )

        if merged is not None:
            all_station_frames.append(merged)

    if not all_station_frames:
        raise ValueError("لم يتم استخراج أي بيانات من ملف Excel.")

    data = pd.concat(all_station_frames, ignore_index=True)
    data = data.sort_values(["station", "year", "month"]).reset_index(drop=True)
    data["date"] = pd.to_datetime(
        dict(year=data["year"], month=data["month"], day=1), errors="coerce"
    )
    discovery = pd.DataFrame(discovery_rows)
    return data, discovery, warnings_list


def build_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [c for c in [*BASE_CLIMATE_FEATURES, "evaporation", "eto"] if c in data.columns]
    for station, group in data.groupby("station"):
        for column in columns:
            valid = int(group[column].notna().sum())
            total = int(len(group))
            rows.append(
                {
                    "station": station,
                    "variable": column,
                    "valid_count": valid,
                    "missing_count": total - valid,
                    "coverage_percent": round(100 * valid / total, 2) if total else 0.0,
                    "minimum": group[column].min(skipna=True),
                    "maximum": group[column].max(skipna=True),
                    "mean": group[column].mean(skipna=True),
                    "std": group[column].std(skipna=True),
                }
            )
    return pd.DataFrame(rows)


def filter_and_engineer_data(data: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    df = data.loc[data["year"].between(config.start_year, config.end_year)].copy()
    if config.target not in df.columns:
        raise ValueError(f"المتغير المستهدف {config.target!r} غير موجود في البيانات.")

    # حذف الصفوف التي لا تحتوي على الهدف أو تحتوي قيمًا غير منطقية.
    df[config.target] = pd.to_numeric(df[config.target], errors="coerce")
    df = df.loc[df[config.target].notna() & (df[config.target] > 0)].copy()

    if config.remove_target_outliers and not df.empty:
        # معالجة محافظة: حذف القيم المتطرفة داخل كل محطة وشهر بواسطة IQR.
        keep = pd.Series(True, index=df.index)
        for _, group in df.groupby(["station", "month"]):
            if len(group) < 8:
                continue
            q1 = group[config.target].quantile(0.25)
            q3 = group[config.target].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 2.5 * iqr, q3 + 2.5 * iqr
            keep.loc[group.index] = group[config.target].between(lower, upper)
        df = df.loc[keep].copy()

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df = df.sort_values(["date", "station"]).reset_index(drop=True)
    return df


def choose_features(df: pd.DataFrame, config: RunConfig) -> list[str]:
    selected = []
    for feature in BASE_CLIMATE_FEATURES:
        if feature not in df.columns:
            continue
        coverage = float(df[feature].notna().mean())
        if coverage >= config.min_feature_coverage:
            selected.append(feature)
    selected.extend(["month_sin", "month_cos"])
    if len(selected) <= 2:
        raise ValueError("لا توجد متغيرات مناخية كافية بعد فحص نسبة التغطية.")
    return selected


def chronological_split(df: pd.DataFrame, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) < max(8, test_years + 4):
        raise ValueError("عدد السنوات غير كافٍ لتقسيم زمني موثوق.")
    test_years_values = years[-test_years:]
    train_df = df.loc[~df["year"].isin(test_years_values)].copy()
    test_df = df.loc[df["year"].isin(test_years_values)].copy()
    return train_df, test_df


def expanding_year_cv(train_df: pd.DataFrame, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """تقسيم تحقق متسلسل يضمن عدم انتقال بيانات سنة الاختبار إلى التدريب."""
    years = np.array(sorted(train_df["year"].unique()))
    min_train_years = max(6, len(years) // 2)
    future_years = years[min_train_years:]
    if len(future_years) < 2:
        raise ValueError("سنوات التدريب غير كافية لبناء Time-Series CV.")

    groups = [g for g in np.array_split(future_years, min(n_splits, len(future_years))) if len(g)]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for group in groups:
        validation_start = int(group[0])
        train_idx = np.flatnonzero(train_df["year"].to_numpy() < validation_start)
        validation_idx = np.flatnonzero(train_df["year"].isin(group).to_numpy())
        if len(train_idx) and len(validation_idx):
            splits.append((train_idx, validation_idx))
    return splits


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    station_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("station", station_pipeline, ["station"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(model: Any, numeric_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features)),
            ("model", model),
        ]
    )


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan, "NSE": np.nan}

    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    nonzero = np.abs(yt) > 1e-12
    mape = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100) if nonzero.any() else np.nan
    r2 = r2_score(yt, yp) if len(yt) > 1 else np.nan
    denominator = np.sum((yt - np.mean(yt)) ** 2)
    nse = 1 - np.sum((yt - yp) ** 2) / denominator if denominator > 0 else np.nan
    return {"n": len(yt), "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "NSE": nse}


def seasonal_climatology_predictions(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> np.ndarray:
    station_month = train_df.groupby(["station", "month"])[target].mean()
    month_only = train_df.groupby("month")[target].mean()
    global_mean = float(train_df[target].mean())
    predictions = []
    for row in test_df.itertuples(index=False):
        value = station_month.get((row.station, row.month), np.nan)
        if pd.isna(value):
            value = month_only.get(row.month, global_mean)
        predictions.append(float(value))
    return np.asarray(predictions)


def persistence_predictions(
    full_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> np.ndarray:
    lookup = full_df.set_index(["station", "year", "month"])[target].to_dict()
    fallback = seasonal_climatology_predictions(train_df, test_df, target)
    predictions = []
    for fallback_value, row in zip(fallback, test_df.itertuples(index=False)):
        previous = lookup.get((row.station, row.year - 1, row.month), np.nan)
        predictions.append(float(previous) if pd.notna(previous) else float(fallback_value))
    return np.asarray(predictions)


def fit_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    features: list[str],
    config: RunConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    input_columns = [*features, "station"]
    X_train = train_df[input_columns]
    y_train = train_df[config.target]
    X_test = test_df[input_columns]
    y_test = test_df[config.target].to_numpy()

    predictions: dict[str, np.ndarray] = {
        "Seasonal Climatology": seasonal_climatology_predictions(train_df, test_df, config.target),
        "Persistence t-12": persistence_predictions(full_df, train_df, test_df, config.target),
    }
    fitted: dict[str, Any] = {}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300 if not config.quick else 120,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250 if not config.quick else 100,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=3,
            random_state=config.random_state,
            loss="huber",
        ),
    }

    for name, model in models.items():
        pipeline = make_pipeline(model, features)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        predictions[name] = pipeline.predict(X_test)

    # SVR مع ضبط المعاملات زمنيًا.
    svr_pipeline = make_pipeline(SVR(kernel="rbf"), features)
    if config.skip_tuning:
        best_svr = clone(svr_pipeline).set_params(
            model__C=100.0, model__gamma="scale", model__epsilon=0.1
        )
        best_svr.fit(X_train, y_train)
        best_params = {
            "model__C": 100.0,
            "model__gamma": "scale",
            "model__epsilon": 0.1,
            "tuning_skipped": True,
        }
        cv_results = pd.DataFrame()
    else:
        cv = expanding_year_cv(train_df, n_splits=3 if config.quick else 4)
        param_grid = {
            "model__C": [10, 100, 500] if config.quick else [1, 10, 100, 500],
            "model__gamma": ["scale", 0.01, 0.1] if config.quick else ["scale", "auto", 0.001, 0.01, 0.1],
            "model__epsilon": [0.05, 0.1, 0.2] if config.quick else [0.01, 0.05, 0.1, 0.2],
        }
        search = GridSearchCV(
            estimator=svr_pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_svr = search.best_estimator_
        best_params = {**search.best_params_, "best_cv_rmse": -float(search.best_score_)}
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")

    fitted["SVR-RBF"] = best_svr
    predictions["SVR-RBF"] = best_svr.predict(X_test)

    prediction_table = test_df[["station", "year", "month", "date", config.target]].copy()
    prediction_table = prediction_table.rename(columns={config.target: "observed"})
    for name, values in predictions.items():
        prediction_table[name] = values
        prediction_table[f"abs_error_{name}"] = np.abs(prediction_table["observed"] - values)

    extra = {
        "best_svr_params": best_params,
        "cv_results": cv_results,
        "input_columns": input_columns,
    }
    return fitted, prediction_table, extra


def create_metric_tables(
    prediction_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = {"station", "year", "month", "date", "observed"}
    model_names = [
        column for column in prediction_table.columns
        if column not in metadata and not column.startswith("abs_error_")
    ]

    overall_rows = []
    station_rows = []
    month_rows = []
    for model in model_names:
        overall_rows.append({"model": model, **metric_dict(prediction_table["observed"], prediction_table[model])})
        for station, group in prediction_table.groupby("station"):
            station_rows.append(
                {"model": model, "station": station, **metric_dict(group["observed"], group[model])}
            )
        for month, group in prediction_table.groupby("month"):
            month_rows.append(
                {
                    "model": model,
                    "month": int(month),
                    "month_name": MONTH_NAMES[int(month)],
                    **metric_dict(group["observed"], group[model]),
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values("RMSE").reset_index(drop=True)
    by_station = pd.DataFrame(station_rows).sort_values(["station", "RMSE"])
    by_month = pd.DataFrame(month_rows).sort_values(["month", "RMSE"])
    return overall, by_station, by_month


def statistical_comparison(prediction_table: pd.DataFrame, reference_model: str = "SVR-RBF") -> pd.DataFrame:
    rows = []
    reference_error = np.abs(
        prediction_table["observed"].to_numpy() - prediction_table[reference_model].to_numpy()
    )
    model_names = [
        column for column in prediction_table.columns
        if column not in {"station", "year", "month", "date", "observed", reference_model}
        and not column.startswith("abs_error_")
    ]

    for model in model_names:
        competitor_error = np.abs(
            prediction_table["observed"].to_numpy() - prediction_table[model].to_numpy()
        )
        difference = reference_error - competitor_error
        try:
            statistic, p_value = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "reference_model": reference_model,
                "competitor": model,
                "mean_absolute_error_difference": float(np.mean(difference)),
                "median_absolute_error_difference": float(np.median(difference)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "significant_at_0_05": bool(p_value < 0.05) if pd.notna(p_value) else False,
                "interpretation": (
                    "SVR lower error" if np.mean(difference) < 0 else "Competitor lower error"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def run_ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_features: list[str],
    best_params: dict[str, Any],
    target: str,
) -> pd.DataFrame:
    climate = [f for f in all_features if f not in {"month_sin", "month_cos"}]
    temperature = [f for f in ["tmean", "tmin", "tmax"] if f in climate]
    scenarios = {
        "Full": all_features,
        "Temperature only": [*temperature, "month_sin", "month_cos"],
        "Temperature + RH + Wind": [
            *temperature,
            *[f for f in ["rh", "wind"] if f in climate],
            "month_sin", "month_cos",
        ],
        "Without solar variables": [
            f for f in all_features if f not in {"sunshine", "solar_rad"}
        ],
        "Without rainfall": [f for f in all_features if f != "rainfall"],
    }

    C = float(best_params.get("model__C", 100.0))
    gamma = best_params.get("model__gamma", "scale")
    epsilon = float(best_params.get("model__epsilon", 0.1))
    rows = []

    for scenario, features in scenarios.items():
        features = list(dict.fromkeys(features))
        if not features or not any(f in climate for f in features):
            continue
        pipeline = make_pipeline(SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon), features)
        input_cols = [*features, "station"]
        pipeline.fit(train_df[input_cols], train_df[target])
        pred = pipeline.predict(test_df[input_cols])
        rows.append(
            {
                "scenario": scenario,
                "features": ", ".join(features),
                **metric_dict(test_df[target], pred),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def run_spatial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    best_params: dict[str, Any],
    target: str,
) -> pd.DataFrame:
    C = float(best_params.get("model__C", 100.0))
    gamma = best_params.get("model__gamma", "scale")
    epsilon = float(best_params.get("model__epsilon", 0.1))
    rows = []

    for held_station in sorted(test_df["station"].unique()):
        spatial_train = train_df.loc[train_df["station"] != held_station]
        spatial_test = test_df.loc[test_df["station"] == held_station]
        if spatial_train.empty or spatial_test.empty:
            continue
        pipeline = make_pipeline(SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon), features)
        cols = [*features, "station"]
        pipeline.fit(spatial_train[cols], spatial_train[target])
        pred = pipeline.predict(spatial_test[cols])
        rows.append(
            {
                "held_out_station": held_station,
                "training_stations": ", ".join(sorted(spatial_train["station"].unique())),
                "train_rows": len(spatial_train),
                "test_rows": len(spatial_test),
                **metric_dict(spatial_test[target], pred),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def compute_permutation_importance_table(
    model: Pipeline,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int,
) -> pd.DataFrame:
    input_cols = [*features, "station"]
    result = permutation_importance(
        model,
        test_df[input_cols],
        test_df[target],
        scoring="neg_root_mean_squared_error",
        n_repeats=15,
        random_state=random_state,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": input_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    table["feature_ar"] = table["feature"].map(FEATURE_LABELS_AR).fillna(table["feature"])
    return table.reset_index(drop=True)


def save_plots(
    prediction_table: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    station_metrics: pd.DataFrame,
    month_metrics: pd.DataFrame,
    importance_table: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) مقارنة RMSE بين النماذج.
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = overall_metrics.sort_values("RMSE")
    ax.bar(ordered["model"], ordered["RMSE"])
    ax.set_title("Model Comparison by RMSE")
    ax.set_ylabel("RMSE (mm/month)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_rmse.png", dpi=220)
    plt.close(fig)

    # 2) مقارنة القيم المقاسة والمتوقعة لكل محطة.
    for station, group in prediction_table.groupby("station"):
        group = group.sort_values("date")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(group["date"], group["observed"], label="Observed")
        ax.plot(group["date"], group["SVR-RBF"], label="SVR-RBF")
        ax.set_title(f"Observed vs SVR-RBF - {station}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Evaporation (mm/month)")
        ax.legend()
        fig.tight_layout()
        safe_station = re.sub(r"[^\w\-]+", "_", str(station), flags=re.UNICODE)
        fig.savefig(figures_dir / f"observed_vs_svr_{safe_station}.png", dpi=220)
        plt.close(fig)

    # 3) Scatter plot.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(prediction_table["observed"], prediction_table["SVR-RBF"], alpha=0.7)
    lower = min(prediction_table["observed"].min(), prediction_table["SVR-RBF"].min())
    upper = max(prediction_table["observed"].max(), prediction_table["SVR-RBF"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_title("Observed vs Predicted Evaporation (SVR-RBF)")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(figures_dir / "svr_scatter.png", dpi=220)
    plt.close(fig)

    # 4) RMSE الشهري لنموذج SVR.
    monthly_svr = month_metrics.loc[month_metrics["model"] == "SVR-RBF"].sort_values("month")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(monthly_svr["month_name"], monthly_svr["RMSE"])
    ax.set_title("Monthly SVR-RBF RMSE")
    ax.set_ylabel("RMSE (mm/month)")
    fig.tight_layout()
    fig.savefig(figures_dir / "svr_monthly_rmse.png", dpi=220)
    plt.close(fig)

    # 5) RMSE حسب المحطة.
    station_svr = station_metrics.loc[station_metrics["model"] == "SVR-RBF"].sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(station_svr["station"], station_svr["RMSE"])
    ax.set_title("SVR-RBF RMSE by Station")
    ax.set_ylabel("RMSE (mm/month)")
    fig.tight_layout()
    fig.savefig(figures_dir / "svr_station_rmse.png", dpi=220)
    plt.close(fig)

    # 6) أهمية المتغيرات.
    fig, ax = plt.subplots(figsize=(9, 6))
    importance_plot = importance_table.sort_values("importance_mean")
    ax.barh(importance_plot["feature_ar"], importance_plot["importance_mean"])
    ax.set_title("Permutation Feature Importance - SVR-RBF")
    ax.set_xlabel("Increase in RMSE after permutation")
    fig.tight_layout()
    fig.savefig(figures_dir / "svr_permutation_importance.png", dpi=220)
    plt.close(fig)


def export_results(
    config: RunConfig,
    clean_data: pd.DataFrame,
    discovery: pd.DataFrame,
    quality: pd.DataFrame,
    prediction_table: pd.DataFrame,
    overall: pd.DataFrame,
    by_station: pd.DataFrame,
    by_month: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    ablation: pd.DataFrame,
    spatial: pd.DataFrame,
    importance: pd.DataFrame,
    cv_results: pd.DataFrame,
    warnings_list: list[str],
    best_params: dict[str, Any],
    selected_features: list[str],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = config.output_dir / "tables"
    models_dir = config.output_dir / "models"
    figures_dir = config.output_dir / "figures"
    tables_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    tables = {
        "cleaned_monthly_data": clean_data,
        "section_discovery": discovery,
        "data_quality": quality,
        "predictions": prediction_table,
        "model_metrics": overall,
        "metrics_by_station": by_station,
        "metrics_by_month": by_month,
        "statistical_tests": statistical_tests,
        "ablation_results": ablation,
        "spatial_validation": spatial,
        "feature_importance": importance,
        "svr_cv_results": cv_results,
    }
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    excel_path = config.output_dir / "Diyala_Evaporation_Results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            # Excel sheet names are limited to 31 characters.
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    run_summary = {
        "excel_input": str(config.excel_path),
        "study_period": [config.start_year, config.end_year],
        "test_years": config.test_years,
        "target": config.target,
        "selected_features": selected_features,
        "best_svr_params": best_params,
        "rows_after_cleaning": len(clean_data),
        "stations": sorted(clean_data["station"].unique().tolist()),
        "warnings": warnings_list,
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (config.output_dir / "warnings.txt").write_text(
        "\n".join(f"- {warning}" for warning in warnings_list) if warnings_list else "لا توجد تحذيرات.",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="تنبؤ التبخر الشهري في ديالى باستخدام SVR ومقارنته مع نماذج أخرى."
    )
    parser.add_argument("--excel", default="data/موحد.xlsx", help="مسار ملف Excel")
    parser.add_argument("--output", default="outputs", help="مجلد النتائج")
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=2016)
    parser.add_argument("--test-years", type=int, default=5)
    parser.add_argument("--target", choices=["evaporation", "eto"], default="evaporation")
    parser.add_argument("--min-coverage", type=float, default=0.65)
    parser.add_argument("--remove-target-outliers", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--quick", action="store_true", help="شبكة معاملات أصغر للتجربة السريعة")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = RunConfig(
        excel_path=Path(args.excel).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        start_year=args.start_year,
        end_year=args.end_year,
        test_years=args.test_years,
        target=args.target,
        min_feature_coverage=args.min_coverage,
        remove_target_outliers=args.remove_target_outliers,
        skip_tuning=args.skip_tuning,
        quick=args.quick,
    )

    if not config.excel_path.exists():
        print(f"خطأ: ملف Excel غير موجود: {config.excel_path}", file=sys.stderr)
        return 2
    if config.start_year > config.end_year:
        print("خطأ: سنة البداية أكبر من سنة النهاية.", file=sys.stderr)
        return 2

    print("[1/9] استخراج جداول ملف Excel...")
    raw_data, discovery, extraction_warnings = extract_workbook(config.excel_path)
    quality_full = build_quality_report(raw_data)

    print("[2/9] تنظيف البيانات وإنشاء المتغيرات الموسمية...")
    clean_data = filter_and_engineer_data(raw_data, config)
    selected_features = choose_features(clean_data, config)
    print(f"      السجلات: {len(clean_data):,} | المحطات: {clean_data['station'].nunique()}")
    print(f"      المتغيرات: {', '.join(selected_features)}")

    train_df, test_df = chronological_split(clean_data, config.test_years)
    train_years = (int(train_df["year"].min()), int(train_df["year"].max()))
    test_years = (int(test_df["year"].min()), int(test_df["year"].max()))
    print(f"[3/9] التقسيم الزمني: تدريب {train_years[0]}–{train_years[1]}، اختبار {test_years[0]}–{test_years[1]}")

    print("[4/9] تدريب النماذج وضبط SVR...")
    fitted, predictions, extra = fit_models(
        train_df, test_df, clean_data, selected_features, config
    )

    print("[5/9] حساب المقاييس والمقارنات الإحصائية...")
    overall, by_station, by_month = create_metric_tables(predictions)
    statistical_tests = statistical_comparison(predictions)

    print("[6/9] تنفيذ Ablation Study...")
    ablation = run_ablation_study(
        train_df, test_df, selected_features, extra["best_svr_params"], config.target
    )

    print("[7/9] تنفيذ التحقق المكاني بين المحطات...")
    spatial = run_spatial_validation(
        train_df, test_df, selected_features, extra["best_svr_params"], config.target
    )

    print("[8/9] حساب أهمية المتغيرات وحفظ النموذج والرسوم...")
    importance = compute_permutation_importance_table(
        fitted["SVR-RBF"], test_df, selected_features, config.target, config.random_state
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = config.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": fitted["SVR-RBF"],
            "features": selected_features,
            "target": config.target,
            "study_period": [config.start_year, config.end_year],
            "best_params": extra["best_svr_params"],
        },
        models_dir / "svr_evaporation_model.joblib",
    )
    save_plots(
        predictions,
        overall,
        by_station,
        by_month,
        importance,
        config.output_dir / "figures",
    )

    warnings_list = list(dict.fromkeys(extraction_warnings))
    if config.end_year >= 2017 and "خانقين" in clean_data["station"].unique():
        warnings_list.append(
            "قيم تبخر خانقين لعامي 2017–2018 أقل بوضوح من السنوات السابقة؛ راجع المصدر الأصلي."
        )

    print("[9/9] تصدير النتائج...")
    export_results(
        config=config,
        clean_data=clean_data,
        discovery=discovery,
        quality=quality_full,
        prediction_table=predictions,
        overall=overall,
        by_station=by_station,
        by_month=by_month,
        statistical_tests=statistical_tests,
        ablation=ablation,
        spatial=spatial,
        importance=importance,
        cv_results=extra["cv_results"],
        warnings_list=warnings_list,
        best_params=extra["best_svr_params"],
        selected_features=selected_features,
    )

    print("\nأفضل النتائج حسب RMSE:")
    print(overall[["model", "MAE", "RMSE", "MAPE", "R2"]].to_string(index=False))
    print("\nأفضل معاملات SVR:")
    print(json.dumps(extra["best_svr_params"], ensure_ascii=False, indent=2, default=str))
    print(f"\nاكتمل التنفيذ. النتائج في: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
