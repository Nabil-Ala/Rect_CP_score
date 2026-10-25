from pathlib import Path
import numpy as np
import pandas as pd


from evaluation import evaluate_cp as _evaluate_cp


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Simulated_datasets"
RESULTS_DIR = BASE_DIR / "results"


def load_npz_dataset(path):
    with np.load(path) as data:
        return data["X"], data["Y"]


def evaluate_cp(X, y, alpha, seed, n_components, dataset_name="dataset",
                output_dir=None):
    return _evaluate_cp(
        X, y, alpha, seed, n_components, dataset_name, max_samples=5000,
    )


def run_experiments(dataset_paths, seeds, alpha=0.1, n_components=3,
                    output_csv=RESULTS_DIR / "results.csv"):
    all_results = []
    seeds = tuple(seeds)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # The first completed evaluation starts a fresh CSV.
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.stem
        X, Y = load_npz_dataset(dataset_path)
        for seed in seeds:
            print(f"running {dataset_name}, seed={seed}", flush=True)
            df = evaluate_cp(
                X, Y, alpha, seed, n_components, dataset_name=dataset_name,
            )
            first = not all_results
            df.to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
            all_results.append(df)

    if not all_results:
        raise ValueError("Provide at least one dataset and one seed.")
    return pd.concat(all_results, ignore_index=True)


def run_full(dataset_dir=DATASET_DIR, seeds=range(20), alpha=0.1,
             n_components=3, output_csv=RESULTS_DIR / "full_results.csv"):
    dataset_dir = Path(dataset_dir)
    dataset_paths = sorted(dataset_dir.glob("*.npz"))
    print("Looking in:", dataset_dir.resolve())
    print("Found", len(dataset_paths), "datasets")
    if not dataset_paths:
        raise ValueError(f"No .npz datasets found in {dataset_dir.resolve()}.")
    return run_experiments(
        dataset_paths, seeds, alpha=alpha, n_components=n_components,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    full_df = run_full()
    print("\nFull run done.")
    print(full_df.head())
