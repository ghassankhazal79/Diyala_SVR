@echo off
chcp 65001 > nul
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --excel "data\موحد.xlsx" --output outputs --start-year 1994 --end-year 2016 --test-years 5
pause
