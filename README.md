# Rectangularization of Multivariate Nonconformity Scores

This repository studies the impact of rectangularization on  nonconformity scores in multivariate conformal prediction. We compare three nonconformity scores (Euclidean, Mahalanobis, and PCP) and their rectangularized versions on simulated and real-world datasets.

We evaluate marginal coverage, conditional coverage metrics, and median log-volume.

## Running the experiments

Use Python 3.11 and run the following commands from the repository root.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run the simulated experiments:
```bash
python main_simu.py
```

Run the real-world experiments:
```bash
python main_real_data.py
```


Results are saved to `results/full_results.csv` and `results/full_real_results.csv`, respectively. 
