python run_mot20_val.py --seq MOT20-01 --fps 25 --wx 0.1 --wy 0.2 --a 10 --cdt 30   --hp --cmc
python run_mot20_val.py --seq MOT20-02 --fps 25 --wx 0.5 --wy 0.5 --a 10 --cdt 30   --hp --cmc
python run_mot20_val.py --seq MOT20-03 --fps 25 --wx 0.1 --wy 3.0 --a 10 --cdt 10   --hp --cmc
python run_mot20_val.py --seq MOT20-05 --fps 25 --wx 0.5 --wy 1.0 --a 10 --cdt 30   --hp --cmc
python eval_mot20.py
pause