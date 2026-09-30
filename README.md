# investing
my own investing tester
print('hello world')

## Usage (pipeline in evaluation_test.py)

```
pip install -r requirements.txt          # + requirements-dev.txt for pytest
```

Price CSVs: columns `time,open,high,low,close` (time = date string or UNIX
seconds/ms), one file per ticker, e.g. `dane_cenowe/BTC-USD.csv` next to the
code, or pass `data_dir=` / `file_path=` (or set `TPI_DATA_DIR`).

```python
import evaluation_test as et

common = dict(ticker='BTC-USD', data_dir='/path/to/csvs', mode='long_only',
              table='main', components=['supertrend'],       # None = all components
              start='2018-01-01', in_sample_end='2025-03-31')
fronts = et.eval(12000, 'crypto', 0.0, n_trials=300, n_jobs=1, seed=42, **common)
robust = et.parameter_robustness_test(12000, 'crypto', et.dedupe_fronts(fronts), 0.0, **common)
for c in robust:
    et.holdout_report(12000, 'crypto', c['params'], '2025-03-31', 0.0, 'long_only', 'main',
                      c['components'], ticker='BTC-USD', data_dir='/path/to/csvs')
```

Choosing time-coherent components: `coherence.coherence_report(price, params, components)`
(or `python coherence.py --ticker BTC-USD --data-dir ... --csv out.csv` for textbook params).

Tests: `python -m pytest tests`. `evaluation.py` is deprecated; use `evaluation_test.py`.
