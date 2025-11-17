# ================================================================
#  GENERALIZED HEALTH DATA PIPELINE (FIXED VERSION)
# ================================================================

import os
import pandas as pd
import numpy as np
import kagglehub

from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore")


# ================================================================
#  LOAD ANY DATASET
# ================================================================

def load_any_dataset(path_or_name):
    if os.path.exists(path_or_name):
        if path_or_name.endswith(".csv"):
            print(f"Loading CSV: {path_or_name}")
            return pd.read_csv(path_or_name)
        if path_or_name.endswith(".xlsx"):
            print(f"Loading XLSX: {path_or_name}")
            return pd.read_excel(path_or_name)
        raise ValueError("Unsupported file type.")

    print(f"Downloading Kaggle dataset: {path_or_name}")
    path = kagglehub.dataset_download(path_or_name)

    for f in os.listdir(path):
        if f.endswith(".csv"):
            print(f"Found CSV: {f}")
            return pd.read_csv(os.path.join(path, f))
        if f.endswith(".xlsx"):
            print(f"Found XLSX: {f}")
            return pd.read_excel(os.path.join(path, f))

    raise ValueError("No CSV or XLSX found in dataset.")


# ================================================================
#  SCHEMA INFERENCE
# ================================================================

def infer_schema(df):
    schema = {"numeric_continuous": [], "categorical_string": [], "binary": [], "id_columns": []}

    for col in df.columns:
        series = df[col]

        # ID
        if series.nunique() == len(series):
            schema["id_columns"].append(col)
            continue

        # Binary
        if series.dropna().isin([0,1,"0","1","e","p","E","P",True,False,"True","False"]).all():
            schema["binary"].append(col)
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            schema["numeric_continuous"].append(col)
            continue

        # String category
        schema["categorical_string"].append(col)

    return schema


# ================================================================
#  BUCKET NUMERIC
# ================================================================

def auto_bucket_numeric(df, columns):
    for col in columns:
        try:
            df[col+"_bucket"] = pd.qcut(df[col], q=4, duplicates="drop")
        except:
            pass
    return df


# ================================================================
#  FIXED ONE-HOT ENCODER (target safe)
# ================================================================

def encode_categoricals(df, schema, target_col):
    cat_cols = [c for c in schema["categorical_string"] + schema["binary"] if c != target_col]

    encoded = pd.get_dummies(df[cat_cols], drop_first=False)

    bucket_cols = [c for c in df.columns if c.endswith("_bucket")]

    final = pd.concat([encoded, df[bucket_cols], df[[target_col]]], axis=1)

    return final.astype(bool)


# ================================================================
#  RULE MINING
# ================================================================

def mine_rules(df):
    print("Mining rules...")
    freq = fpgrowth(df, min_support=0.5, max_len=2, use_colnames=True)

    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    if rules.empty:
        return rules

    rules["weight"] = rules["confidence"] * np.log(1 + rules["consequent support"])
    rules["antecedents_set"] = rules["antecedents"].apply(lambda x: set(x))
    return rules


# ================================================================
#  APPLY RULES
# ================================================================

def compute_risk_score(rules, record):
    features = set(idx for idx, val in record.items() if val == 1)
    applicable = [row for _, row in rules.iterrows() if row["antecedents_set"].issubset(features)]

    if not applicable:
        return 0.0, [], 0.0

    total_weight = sum(r["weight"] for r in applicable)
    normalized = [(f"Rule_{i}", r["weight"] / total_weight) for i, r in enumerate(applicable)]
    score = sum(w * r["confidence"] for (_, w), r in zip(normalized, applicable))
    return total_weight, normalized, score


def apply_rules_to_dataset(df, rules):
    tw_list, nw_list, rs_list = [], [], []

    for _, row in df.iterrows():
        tw, nw, rs = compute_risk_score(rules, row)
        tw_list.append(tw)
        nw_list.append(nw)
        rs_list.append(rs)

    df = df.copy()
    df["Total_Rule_Weight"] = tw_list
    df["Rule_Risk_Score"] = rs_list
    df["Normalized_Rule_Weights"] = nw_list
    return df


def expand_normalized(df):
    expanded = pd.DataFrame()

    for idx, row in df.iterrows():
        for rule_id, weight in row["Normalized_Rule_Weights"]:
            expanded.loc[idx, rule_id] = weight

    expanded.fillna(0, inplace=True)
    return pd.concat([df.drop(columns=["Normalized_Rule_Weights"]), expanded], axis=1)


# ================================================================
#  TARGET SELECTION
# ================================================================

def pick_target_column(df):
    print("\n=== Select Target Column ===")
    candidates = []

    for col in df.columns:
        if df[col].nunique() == 2:
            candidates.append(col)

    for idx, c in enumerate(candidates, start=1):
        print(f"{idx}. {c} (binary)")

    ch = int(input("Pick target: "))
    return candidates[ch-1], "binary"


# ================================================================
#  TRAIN NN
# ================================================================

def train_nn(df, target_name, target_type, model_name):
    y_raw = df[target_name].astype(str)
    X = df.drop(columns=[target_name])

    y = LabelEncoder().fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, validation_split=0.2)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Final Accuracy: {acc:.4f}")

    model.save(model_name)
    print("Saved:", model_name)


# ================================================================
#  PIPELINE
# ================================================================

def run_pipeline():
    print("\n=== GENERALIZED PIPELINE (FIXED) ===\n")

    dataset = input("Dataset path or Kaggle name: ")
    df = load_any_dataset(dataset)

    schema = infer_schema(df)

    # Pick target BEFORE encoding
    target_col, target_type = pick_target_column(df)

    df = auto_bucket_numeric(df.copy(), schema["numeric_continuous"])
    df_encoded = encode_categoricals(df, schema, target_col)

    # Rule mining
    rules = mine_rules(df_encoded.astype(bool))
    
    df_rules = apply_rules_to_dataset(df_encoded, rules)
    df_final = expand_normalized(df_rules)

    df_final.to_csv("generalized_rule_enhanced.csv", index=False)
    print("Saved: generalized_rule_enhanced.csv")

    model_name = input("Model filename (e.g., something.keras): ")
    train_nn(df_final, target_col, target_type, model_name)


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    run_pipeline()