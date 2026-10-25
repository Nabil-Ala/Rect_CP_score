
from pathlib import Path
import pandas as pd


from data.data_loading import load_real_data
from evaluation import evaluate_cp as _evaluate_cp


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
COMPONENTS_BY_DATASET = {
    "households": 3, "air": 3, "bio": 3, "births2": 3,
    "rf1": 3, "taxi": 3, "rf2": 3,
}


def evaluate_cp(X, y, alpha, seed, n_components, dataset_name="dataset",
                output_dir=None):
    return _evaluate_cp(
        X, y, alpha, seed, n_components, dataset_name, max_samples=8000,
    )


def run_real_experiments(dataset_specs, seeds=range(2), alpha=0.1,
                         n_components=3, output_csv=RESULTS_DIR / "real_results.csv"):
    """Relative data roots are resolved against this script's directory."""
    all_results = []
    seeds = tuple(seeds)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    for root, family, name in dataset_specs:
        print(f"\n=== dataset: {family}/{name} ===")
        root = Path(root)
        if not root.is_absolute():
            root = BASE_DIR / root
        X, Y = load_real_data(root, family, name)
        #X,Y = X[:2000], Y[:2000]
        components = COMPONENTS_BY_DATASET.get(name.lower(), n_components)
        dataset_name = f"{family}_{name}"
        for seed in seeds:
            print(f"running {dataset_name}, seed={seed}", flush=True)
            df = evaluate_cp(
                X, Y, alpha, seed, components, dataset_name=dataset_name,
            )
            first = not all_results
            df.to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
            all_results.append(df)

    if not all_results:
        raise ValueError("Provide at least one dataset and one seed.")
    return pd.concat(all_results, ignore_index=True)


real_specs = [
     ("data", "feldman", "bio"),
     ("data", "cevid", "births2"),
    ("data", "mulan", "rf1"),
    ("data", "camehl", "households"),
    ("data", "cevid", "air"),
    ("data", "wang", "taxi"),
    ("data", "mulan", "rf2"),
]


if __name__ == "__main__":
    real_df = run_real_experiments(
        dataset_specs=real_specs,
        seeds=range(20),
        alpha=0.1,
        n_components=3,
        output_csv=RESULTS_DIR / "full_real_results.csv",
    )
    print("\nReal-data run done.")
    print(real_df.head())
