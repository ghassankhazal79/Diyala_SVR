#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""واجهة Streamlit عربية لتشغيل التحليل وعرض النتائج."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="التنبؤ بالتبخر في ديالى", layout="wide")
st.title("نظام التنبؤ الشهري بالتبخر في محافظة ديالى")
st.caption("SVR-RBF مع مقارنة Random Forest وGradient Boosting والانحدار الخطي والنماذج المرجعية")

uploaded = st.file_uploader("ارفع ملف موحد.xlsx", type=["xlsx"])
col1, col2, col3 = st.columns(3)
start_year = col1.number_input("سنة البداية", min_value=1900, max_value=2100, value=1994)
end_year = col2.number_input("سنة النهاية", min_value=1900, max_value=2100, value=2016)
test_years = col3.number_input("عدد سنوات الاختبار", min_value=2, max_value=10, value=5)
quick = st.checkbox("تشغيل سريع", value=True)

if st.button("بدء التحليل", type="primary"):
    if uploaded is None:
        st.error("يرجى رفع ملف Excel أولًا.")
    else:
        work_dir = Path("streamlit_run")
        work_dir.mkdir(exist_ok=True)
        excel_path = work_dir / "موحد.xlsx"
        excel_path.write_bytes(uploaded.getbuffer())
        output_dir = work_dir / "outputs"
        command = [
            sys.executable, "main.py",
            "--excel", str(excel_path),
            "--output", str(output_dir),
            "--start-year", str(int(start_year)),
            "--end-year", str(int(end_year)),
            "--test-years", str(int(test_years)),
        ]
        if quick:
            command.append("--quick")
        with st.spinner("يجري استخراج البيانات وتدريب النماذج..."):
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(result.stderr or result.stdout)
        else:
            st.success("اكتمل التحليل")
            st.code(result.stdout)
            metrics_path = output_dir / "tables" / "model_metrics.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                st.subheader("مقارنة النماذج")
                st.dataframe(metrics, use_container_width=True)
                st.bar_chart(metrics.set_index("model")["RMSE"])
            for image_name in [
                "model_comparison_rmse.png",
                "svr_scatter.png",
                "svr_monthly_rmse.png",
                "svr_station_rmse.png",
                "svr_permutation_importance.png",
            ]:
                image_path = output_dir / "figures" / image_name
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
            excel_result = output_dir / "Diyala_Evaporation_Results.xlsx"
            if excel_result.exists():
                st.download_button(
                    "تحميل ملف النتائج Excel",
                    data=excel_result.read_bytes(),
                    file_name=excel_result.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
