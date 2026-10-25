
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import arff

def load_real_data(data_dir, group, name):
    data_path = Path(data_dir)

    # ----------------------------
    # 1) Load raw data + targets
    # ----------------------------
    if group == "camehl":
        path = data_path / "camehl" / f"{name}.csv"
        df = pd.read_csv(path, index_col=0)
        df = df.drop(columns=["newid", "inc.a"], errors="ignore")
        targets = ["inc", "food", "house", "utili"]

    elif group == "cevid":
        path = data_path / "cevid" / f"{name}.csv"
        df = pd.read_csv(path)
        if name == "air":
            df = df.replace(-200, np.nan)
        cevid_nb_targets = {"air": 6, "births1": 2, "births2": 4, "wage": 2}
        nb_targets = cevid_nb_targets[name]
        targets = df.columns[-nb_targets:].tolist()

    elif group == "mulan":
        name_l = name.lower()

        # Resolve the actual spelling, including the supplied RF2.csv, on all OSes.
        mulan_dir = data_path / "mulan"
        files = {path.name.lower(): path for path in sorted(mulan_dir.iterdir())}
        path_csv = files.get(f"{name_l}.csv", mulan_dir / f"{name_l}.csv")
        path_arff = files.get(f"{name_l}.arff", mulan_dir / f"{name_l}.arff")

        if path_csv.exists():
            df = pd.read_csv(path_csv, na_values=["?", "NA", "NaN", "nan", ""])
            # pandas CSV exports can contain a saved row index, not a predictor.
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        elif path_arff.exists():
            data, meta = arff.loadarff(path_arff)
            df = pd.DataFrame(data)

            # decode byte strings to normal strings
            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].apply(
                    lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
                )
        else:
            raise FileNotFoundError(f"Neither found: {path_csv} nor {path_arff}")

        df.columns = [str(c).strip() for c in df.columns]

        mulan_nb_targets = {
            "wq": 14, "rf1": 8, "rf2": 8, "scm1d": 16, "scm20d": 16,
            "atp1d": 6, "atp7d": 6, "andro": 6,
        }
        if name_l not in mulan_nb_targets:
            raise ValueError(f"Unknown mulan dataset '{name}'. Add it to mulan_nb_targets.")

        nb_targets = mulan_nb_targets[name_l]
        targets = df.columns[-nb_targets:].tolist()


    elif group == "wang":
        path = data_path / "wang" / f"{name}.csv"
        df = pd.read_csv(path)
        wang_nb_targets = {"energy": 2, "taxi": 2}
        nb_targets = wang_nb_targets[name]
        targets = df.columns[-nb_targets:].tolist()

    elif group == "feldman":
        path = data_path / "feldman" / f"{name}.csv"
        if name == "blog_data":
            df = pd.read_csv(path, header=None)
        elif name in ["bio", "house"]:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, index_col=0)

        feldman_targets = {
            "bio": ["F7", "F9"],
            "blog_data": [60, 280],
            "house": ["lat", "price"],
            "meps_19": ["K6SUM42", "UTILIZATION_reg"],
            "meps_20": ["K6SUM42", "UTILIZATION_reg"],
            "meps_21": ["K6SUM42", "UTILIZATION_reg"],
        }
        targets = feldman_targets[name]

    else:
        raise ValueError(f"Unknown group: {group}")

    # ----------------------------
    # 2) Split X/y + coerce y numeric
    # ----------------------------
    y_df = df[targets].apply(pd.to_numeric, errors="coerce")
    x_df = df.drop(columns=targets)

    # drop rows where y became NaN
    keep = ~y_df.isna().any(axis=1)
    x_df = x_df.loc[keep].copy()
    y_df = y_df.loc[keep].copy()

    # ----------------------------
    # 3) Dheur inspired preprocessing http://github.com/Vekteur/multi-output-conformal-regression/blob/6d40a669492ef1c7860568315796a99a895d9548/moc/datamodules/preprocessing.py
    # ----------------------------
    # identify categoricals
    nunique = x_df.nunique(dropna=True)
    cat_mask = (x_df.dtypes == "object") | (x_df.dtypes == "bool")
    cat_mask |= (x_df.dtypes == "int64") & (nunique <= 10)
    cat_mask |= (nunique == 2)

    # remove high-cardinality categoricals (>20 unique)
    high_card = (nunique > 20) & cat_mask
    x_df = x_df.drop(columns=x_df.columns[high_card.to_numpy()], errors="ignore")
    cat_mask = cat_mask.loc[x_df.columns]

    # remove pseudo-categorical numerics: numeric cols with <10 unique values
    num_mask = ~cat_mask
    pseudo = x_df.nunique(dropna=True) < 10
    to_drop = (num_mask & pseudo)
    x_df = x_df.drop(columns=x_df.columns[to_drop.to_numpy()], errors="ignore")
    cat_mask = cat_mask.loc[x_df.columns]

    # remove missing-heavy cols (>20%), then rows with any missing
    miss_cols = x_df.isna().mean(axis=0) > 0.2
    x_df = x_df.drop(columns=x_df.columns[miss_cols.to_numpy()], errors="ignore")
    miss_rows = x_df.isna().any(axis=1)
    x_df = x_df.loc[~miss_rows].copy()
    y_df = y_df.loc[~miss_rows].copy()

    # one-hot encode remaining categoricals
    x_df = pd.get_dummies(x_df)

    # fail fast if too small
    if x_df.shape[1] == 0:
        raise ValueError("No remaining columns after preprocessing.")
    if len(x_df) < 100:
        raise ValueError("Not enough rows after preprocessing (<100).")

    # numpy float32 + safety
    x = x_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)

    if np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("NaN/Inf present after preprocessing.")

    return x, y
