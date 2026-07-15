# برنامج التنبؤ الشهري بالتبخر في محافظة ديالى

يقرأ البرنامج ملف **موحد.xlsx** ويستخرج آليًا جداول الحرارة والرطوبة والرياح والأمطار والسطوع والإشعاع الشمسي والتبخر الحوضي من أوراق خانقين والخالص وبلدروز، ثم يحولها إلى قاعدة بيانات شهرية ويطبق نموذج **RBF-SVR**.

## المقارنات المنفذة

- Seasonal Climatology.
- Persistence باستخدام قيمة الشهر نفسه من السنة السابقة `t-12`.
- Multiple Linear Regression.
- Random Forest.
- Gradient Boosting.
- RBF Support Vector Regression بعد ضبط `C` و`gamma` و`epsilon` بتحقق زمني متوسع.

كما ينفذ البرنامج:

- MAE وRMSE وMAPE وR² وNSE.
- نتائج مستقلة لكل محطة ولكل شهر.
- اختبار Wilcoxon لمقارنة الأخطاء المطلقة.
- Ablation Study لمجموعات المدخلات.
- Leave-One-Station-Out Spatial Validation.
- Permutation Feature Importance.

## التشغيل السريع على Windows

1. ضع الملف داخل المجلد `data` باسم `موحد.xlsx`.
2. شغّل `run_analysis.bat`.
3. تظهر النتائج داخل `outputs`.

## التشغيل من سطر الأوامر

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --excel "data/موحد.xlsx" --output outputs --start-year 1994 --end-year 2016 --test-years 5
```

للتجربة الأسرع:

```bash
python main.py --excel "data/موحد.xlsx" --output outputs --quick
```

## الواجهة العربية

```bash
streamlit run app.py
```

أو شغّل `run_dashboard.bat`.

## التنبؤ ببيانات جديدة

بعد تدريب النموذج:

```bash
python predict.py --model outputs/models/svr_evaporation_model.joblib --input sample_new_data.csv --output new_predictions.csv
```

## بنية النتائج

- `outputs/Diyala_Evaporation_Results.xlsx`: جميع الجداول في ملف واحد.
- `outputs/tables`: ملفات CSV.
- `outputs/figures`: الرسوم.
- `outputs/models/svr_evaporation_model.joblib`: النموذج المدرب.
- `outputs/run_summary.json`: الإعدادات والمعاملات المختارة.
- `outputs/warnings.txt`: تحذيرات جودة البيانات.

## ملاحظات علمية

- الفترة الافتراضية هي **1994–2016** لأنها الفترة المشتركة بين المحطات الثلاث وتستبعد القيم غير المتسقة في تبخر خانقين لعامي 2017–2018.
- الورقة المسماة «بلدروز» تتضمن في أحد العناوين `STATION: BADRA`، لذلك يجب التحقق من هوية المحطة قبل النشر النهائي.
- لا يستخدم البرنامج التبخر–نتح كمدخل عند التنبؤ بالتبخر الحوضي لتجنب تسرب المعلومات.
