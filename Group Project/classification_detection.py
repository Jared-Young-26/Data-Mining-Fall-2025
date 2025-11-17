# ================================================================
#  GENERALIZED HEALTH DATA CLASSIFICATION + RULE MINING PIPELINE
#  Works for ANY dataset (CSV/XLSX/Kaggle) — fully automated.
#  Author: ChatGPT (custom build for Jared Young)
# ================================================================

import os
import math
import pandas as pd
import numpy as np
import kagglehub

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ================================================================
#  LOAD DATA (CSV, XLSX, OR KAGGLE)
# ================================================================

def load_any_dataset(path_or_name):
    """
    Loads:
    - CSV file
    - XLSX file
    - Kaggle dataset name
    """
    if os.path.exists(path_or_name):
        if path_or_name.endswith(".csv"):
            print(f"Loading CSV: {path_or_name}")
            return pd.read_csv(path_or_name)
        if path_or_name.endswith(".xlsx"):
            print(f"Loading XLSX: {path_or_name}")
            return pd.read_excel(path_or_name)
        raise ValueError("Unsupported local file format (use CSV or XLSX).")

    print(f"Attempting Kaggle dataset download: {path_or_name}")
    path = kagglehub.dataset_download(path_or_name)

    # Attempt to detect CSV inside
    for f in os.listdir(path):
        if f.endswith(".csv"):
            print(f"Dataset found: {f}")
            return pd.read_csv(os.path.join(path, f))
        if f.endswith(".xlsx"):
            print(f"Dataset found: {f}")
            return pd.read_excel(os.path.join(path, f))

    raise ValueError("No valid CSV/XLSX found in the Kaggle dataset.")


# ================================================================
#  SCHEMA INFERENCE
# ================================================================

def infer_schema(df):
    schema = {
        "numeric_continuous": [],
        "categorical_string": [],
        "binary": [],
        "id_columns": []
    }

    for col in df.columns:
        series = df[col]

        if series.nunique() == len(series):
            schema["id_columns"].append(col)
            continue

        # Detect binary
        if series.dropna().isin([0, 1, "0", "1", True, False, "Yes", "No"]).all():
            schema["binary"].append(col)
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            schema["numeric_continuous"].append(col)
            continue

        # String/categorical
        schema["categorical_string"].append(col)

    return schema


# ================================================================
#  BUCKETING LOGIC (DYNAMIC)
# ================================================================

def auto_bucket_numeric(df, columns):
    """
    Uses quantile buckets for unknown columns.
    Applies medically meaningful buckets if column name matches patterns.
    """
    for col in columns:
        col_lower = col.lower()

        try:
            # Known medical patterns
            if "bmi" in col_lower:
                df[col+"_bucket"] = pd.cut(df[col],
                                           bins=[0, 18.5, 25, 30, 60],
                                           labels=["Underweight","Normal","Overweight","Obese"])
            elif "glucose" in col_lower:
                df[col+"_bucket"] = pd.cut(df[col],
                                           bins=[0, 100, 140, 200, 500],
                                           labels=["Normal","Prediabetic","Diabetic","Severe"])
            elif "pressure" in col_lower or "bp" in col_lower:
                df[col+"_bucket"] = pd.qcut(df[col], q=4, duplicates="drop")
            else:
                # Fallback: quantiles
                df[col+"_bucket"] = pd.qcut(df[col], q=4, duplicates="drop")
        except Exception:
            continue

    return df


# ================================================================
#  ONE-HOT ENCODING
# ================================================================

def encode_categoricals(df, schema):
    cat_cols = schema["categorical_string"] + schema["binary"]

    encoded = pd.get_dummies(df[cat_cols], drop_first=False)
    numeric_buckets = [c for c in df.columns if c.endswith("_bucket")]

    final_df = pd.concat([encoded, df[numeric_buckets]], axis=1)
    final_df = final_df.astype(bool)

    return final_df


# ================================================================
#  RULE MINING
# ================================================================

def mine_rules(itemset_df):
    print("Running Apriori...")
    freq = apriori(itemset_df, min_support=0.3, use_colnames=True)

    rules = association_rules(freq, metric="confidence", min_threshold=0.5)

    # Compute weights
    weights = []
    for _, row in rules.iterrows():
        weight = row["confidence"] * math.log(1 + row["consequent support"])
        weights.append(weight)

    rules["weight"] = weights
    rules["antecedents_set"] = rules["antecedents"].apply(lambda x: set(x))

    return rules


# ================================================================
#  APPLY RULES TO RECORDS
# ================================================================

def compute_risk_score(rules, record):
    features = set(idx for idx, val in record.items() if val == 1)
    applicable_rules = [rule for _, rule in rules.iterrows() if rule["antecedents_set"].issubset(features)]

    if not applicable_rules:
        return 0.0, [], 0.0

    total_weight = sum(rule["weight"] for rule in applicable_rules)

    normalized = [
        (f"Rule_{i}", rule["weight"] / total_weight)
        for i, rule in enumerate(applicable_rules)
    ]

    score = sum((w * rule["confidence"]) for (rid, w), rule in zip(normalized, applicable_rules))

    return total_weight, normalized, score


def apply_rules_to_dataset(df, rules):
    total_weights = []
    normalized_weights = []
    risk_scores = []

    for idx, row in df.iterrows():
        tw, nw, rs = compute_risk_score(rules, row)
        total_weights.append(tw)
        normalized_weights.append(nw)
        risk_scores.append(rs)

    df = df.copy()
    df["Total_Rule_Weight"] = total_weights
    df["Rule_Risk_Score"] = risk_scores
    df["Normalized_Rule_Weights"] = normalized_weights

    return df


def expand_normalized(df):
    new_cols = pd.DataFrame()

    for idx, row in df.iterrows():
        for rule_id, weight in row["Normalized_Rule_Weights"]:
            new_cols.loc[idx, rule_id] = weight

    new_cols.fillna(0, inplace=True)
    return pd.concat([df.drop(columns=["Normalized_Rule_Weights"]), new_cols], axis=1)


# ================================================================
#  TARGET COLUMN SELECTION
# ================================================================

def pick_target_column(df_original, df_encoded):
    """
    Shows binary and multi-class candidates to user.
    """
    binary_candidates = []
    multiclass_candidates = []

    for col in df_original.columns:
        unique_vals = df_original[col].dropna().unique()
        if len(unique_vals) == 2:
            binary_candidates.append(col)
        elif 2 < len(unique_vals) <= 10:
            multiclass_candidates.append(col)

    print("\n=== Eligible Target Columns ===")
    idx = 1
    mapping = {}

    for c in binary_candidates:
        print(f"{idx}. {c} (binary)")
        mapping[idx] = (c, "binary")
        idx += 1

    for c in multiclass_candidates:
        print(f"{idx}. {c} (multiclass)")
        mapping[idx] = (c, "multi")
        idx += 1

    choice = int(input("Select a target column: "))
    return mapping[choice]


# ================================================================
#  TRAIN NEURAL NETWORK
# ================================================================

def train_nn(df, target_name, target_type, model_name="general_model.keras"):

    # Extract original target
    y_raw = df[target_name]

    # Convert df to X by dropping target and non-numeric columns
    X = df.drop(columns=[c for c in df.columns if c == target_name])

    # Encode y depending on type
    if target_type == "binary":
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        loss = "binary_crossentropy"
        activation = "sigmoid"
        output_units = 1
    else:
        le = LabelEncoder()
        y_indexed = le.fit_transform(y_raw.astype(str))
        y = to_categorical(y_indexed)
        loss = "categorical_crossentropy"
        activation = "softmax"
        output_units = y.shape[1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build model
    model = Sequential([
        Dense(128, activation="relu", input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(output_units, activation=activation)
    ])

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)

    _, acc = model.evaluate(X_test, y_test)
    print(f"\nMODEL ACCURACY: {acc:.4f}\n")

    model.save(model_name)
    print(f"Model saved: {model_name}")

    return model


# ================================================================
#  MAIN PIPELINE
# ================================================================

def run_pipeline():

    print("\n=== GENERALIZED HEALTH DATA PIPELINE ===\n")

    preprocess_choice = input("Run preprocessing? (Y/N): ").strip().lower()
    if preprocess_choice == "y":
        dataset_path = input("Enter dataset (CSV, XLSX, or Kaggle dataset name): ")
        df_original = load_any_dataset(dataset_path)

        schema = infer_schema(df_original)
        df_buckets = auto_bucket_numeric(df_original.copy(), schema["numeric_continuous"])
        df_encoded = encode_categoricals(df_buckets, schema)

        # RULE MINING
        itemsets = df_encoded.astype(bool)
        rules = mine_rules(itemsets)

        df_rules = apply_rules_to_dataset(itemsets, rules)
        df_expanded = expand_normalized(df_rules)

        df_expanded.to_csv("generalized_rule_enhanced.csv", index=False)
        print("\nSaved: generalized_rule_enhanced.csv\n")

    else:
        df_expanded = pd.read_csv("generalized_rule_enhanced.csv")
        df_original = df_expanded.copy()

    # TARGET SELECTION
    target_col, target_type = pick_target_column(df_original, df_expanded)

    print(f"\nSelected target: {target_col} ({target_type})\n")

    # TRAIN MODEL
    model_name = input("Enter model name to save: ")
    train_nn(df_expanded, target_col, target_type, model_name=model_name)

    print("\nPipeline completed.\n")


# ================================================================
#  RUN
# ================================================================

if __name__ == "__main__":
    run_pipeline()
