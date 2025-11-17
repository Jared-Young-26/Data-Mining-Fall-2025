import math
import kagglehub
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

def fetch_dataset():
    # Download latest version
    path = kagglehub.dataset_download("mohankrishnathalla/diabetes-health-indicators-dataset")

    print("Path to dataset files:", path)
    return path

def preprocess_data(path):

    # Load the dataset
    data = pd.read_csv(f"{path}/diabetes_dataset.csv")

    # Age (18–90)
    data['age_group'] = pd.cut(data['age'], 
                            bins=[18, 30, 40, 50, 60, 70, 80, 90],
                            labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-90'])

    # Alcohol consumption per week
    data['alcohol_group'] = pd.cut(data['alcohol_consumption_per_week'],
                                bins=[0, 1, 3, 7, 14, 21, 100],
                                labels=['None', 'Low', 'Moderate', 'Frequent', 'Heavy', 'Extreme'])

    # Physical activity per week
    data['activity_group'] = pd.cut(data['physical_activity_minutes_per_week'],
                                    bins=[0, 60, 120, 180, 300, 600, 1000],
                                    labels=['Sedentary', 'Low', 'Moderate', 'Active', 'VeryActive', 'Athlete'])

    # Diet score
    data['diet_group'] = pd.cut(data['diet_score'], 
                                bins=[0, 3, 5, 7, 8.5, 10],
                                labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Excellent'])

    # Sleep (hours/day)
    data['sleep_group'] = pd.cut(data['sleep_hours_per_day'], 
                                bins=[3, 6.3, 7, 7.7, 10],
                                labels=['Deprivation', 'Poor', 'Healthy', 'Oversleeping'])

    # BMI
    data['bmi_group'] = pd.cut(data['bmi'], 
                                bins=[15, 23.2, 25.6, 28, 39.2],
                                labels=['Underweight', 'Healthy', 'Overweight', 'Obesity'])

    # Waist-to-Hip Ratio
    data['waist_ratio'] = pd.cut(data['waist_to_hip_ratio'],
                                bins=[0.67, 0.82, 0.86, 0.89, 1.06],
                                labels=['Low', 'Moderate', 'High', 'VeryHigh'])


    # --- Cardiovascular Indicators ---

    # Systolic Blood Pressure (mmHg)
    data['systolic_bp'] = pd.cut(data['systolic_bp'], 
                                bins=[80, 100, 120, 130, 140, 160, 200],
                                labels=['Low', 'Ideal', 'Elevated', 'Hypertension_Stage1', 'Hypertension_Stage2', 'Crisis'])

    # Diastolic Blood Pressure (mmHg)
    data['diastolic_bp'] = pd.cut(data['diastolic_bp'], 
                                bins=[40, 60, 80, 90, 100, 120],
                                labels=['Low', 'Ideal', 'Elevated', 'Stage1', 'Stage2'])

    # Heart Rate (bpm)
    data['heart_rate'] = pd.cut(data['heart_rate'],
                                bins=[40, 60, 80, 100, 120, 200],
                                labels=['Bradycardia', 'Normal', 'Elevated', 'Tachycardia', 'Extreme'])

    # --- Lipid Profile ---

    # Total Cholesterol (mg/dL)
    data['cholesterol_level'] = pd.cut(data['cholesterol_total'],
                                    bins=[100, 160, 200, 240, 300, 400],
                                    labels=['Low', 'Desirable', 'Borderline', 'High', 'VeryHigh'])

    # HDL Cholesterol (mg/dL)
    data['hdl_level'] = pd.cut(data['hdl_cholesterol'],
                            bins=[10, 40, 60, 100],
                            labels=['Low', 'Normal', 'High'])

    # LDL Cholesterol (mg/dL)
    data['ldl_level'] = pd.cut(data['ldl_cholesterol'],
                            bins=[30, 100, 130, 160, 190, 300],
                            labels=['Optimal', 'NearOptimal', 'Borderline', 'High', 'VeryHigh'])

    # Triglycerides (mg/dL)
    data['triglycerides_level'] = pd.cut(data['triglycerides'],
                                        bins=[20, 150, 200, 500, 1000],
                                        labels=['Normal', 'Borderline', 'High', 'VeryHigh'])

    # --- Glucose Metabolism ---

    # Fasting Glucose (mg/dL)
    data['glucose_fasting'] = pd.cut(data['glucose_fasting'],
                                    bins=[50, 100, 126, 200, 400],
                                    labels=['Normal', 'Prediabetic', 'Diabetic', 'Severe'])

    # Postprandial Glucose (mg/dL)
    data['glucose_postprandial'] = pd.cut(data['glucose_postprandial'],
                                        bins=[50, 140, 200, 400],
                                        labels=['Normal', 'Prediabetic', 'Diabetic'])

    # Insulin (μU/mL)
    data['insulin_level'] = pd.cut(data['insulin_level'],
                                bins=[1, 5, 20, 40, 100],
                                labels=['Low', 'Normal', 'Elevated', 'Severe'])

    # HbA1c (%)
    data['hba1c_level'] = pd.cut(data['hba1c'],
                                bins=[3.5, 5.7, 6.5, 8, 10, 14],
                                labels=['Normal', 'Prediabetic', 'Diabetic', 'PoorControl', 'Severe'])

    # --- Risk & Stage ---

    # Diabetes Risk Score (0–100)
    data['diabetes_risk_score'] = pd.cut(data['diabetes_risk_score'],
                                        bins=[0, 25, 50, 75, 90, 100],
                                        labels=['Low', 'Mild', 'Moderate', 'High', 'Severe'])

    
    # One-hot encode categorical features
    categorical_features = [
        'gender', 'ethnicity', 'education_level', 'income_level', 
        'employment_status', 'smoking_status',
        'age_group', 'alcohol_group', 'activity_group', 'diet_group',
        'sleep_group', 'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history', 'bmi_group', 'waist_ratio',
        'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_level',
        'hdl_level', 'ldl_level', 'triglycerides_level', 'glucose_fasting',
        'glucose_postprandial', 'insulin_level', 'hba1c_level', 'diabetes_risk_score',
        'diabetes_stage', 'diagnosed_diabetes'
    ]

    processed_data = pd.get_dummies(data[categorical_features])

    # Convert to boolean
    finalized_data = processed_data.astype(bool)
    return finalized_data

def generate_rules(data):
    # Mine Frequent Itemsets with Apriori
    frequent_itemsets = apriori(data, min_support =0.5, use_colnames =True)
    print(frequent_itemsets)

    # Generate Associative Rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(rules)
    return rules

def compute_rule_weights(rules):
    # Calculate weights for each rule based on support and confidence
    weights = []
    for _, row in rules.iterrows():
        weight = row['confidence'] * math.log(1 + row['consequent support'])
        weights.append(weight)
    rules['weight'] = weights
    return rules

def compute_risk_score(applicable_rules, normalized_weights):
    """
    Compute a rule-based risk score for diabetes.
    Score = sum(normalized_weight * rule_confidence for all applicable rules)
    """
    if not applicable_rules:
        return 0.0
    score = 0.0
    for (rule_id, weight), rule in zip(normalized_weights, applicable_rules):
        score += weight * rule["confidence"]
    return score

def process_rules_for_records(rules, records):
    records = records.copy()
    total_weights, normalized_rule_weights, risk_scores = [], [], []

    rules = rules.copy()
    rules["antecedents_set"] = rules["antecedents"].apply(lambda x: set(x))

    for idx, record in records.iterrows():
        applicable_rules = []
        active_features = set(record[record == True].index)

        for _, rule in rules.iterrows():
            if rule["antecedents_set"].issubset(active_features):
                applicable_rules.append(rule)

        total_weight = sum(rule["weight"] for rule in applicable_rules)
        total_weights.append(total_weight)

        if total_weight > 0:
            norm_weights = [
                (f"Rule_{i}", rule["weight"] / total_weight)
                for i, rule in enumerate(applicable_rules)
            ]
        else:
            norm_weights = []

        normalized_rule_weights.append(norm_weights)
        # Compute rule-based risk score
        risk_score = compute_risk_score(applicable_rules, norm_weights)
        risk_scores.append(risk_score)

    records["Total_Rule_Weight"] = total_weights
    records["Normalized_Rule_Weights"] = normalized_rule_weights
    records["Rule_Risk_Score"] = risk_scores

    print("Rule processing complete.")
    print(f"Processed {len(records)} records with {len(rules)} rules.")

    return records

def expand_normalized_rules(records):
    rule_features = pd.DataFrame()

    for idx, row in records.iterrows():
        for rule_id, weight in row['Normalized_Rule_Weights']:
            rule_features.loc[idx, rule_id] = weight

    rule_features.fillna(0, inplace=True)
    return pd.concat([records.drop(columns=['Normalized_Rule_Weights']), rule_features], axis=1)

# Neural Network Classification
def train_nn(expanded_df, model_name="diabetes_nn_model.keras", target_column="diagnosed_diabetes", loss="binary_crossentropy", activation="sigmoid", epochs=20, batch_size=32):
    y = expanded_df[target_column].astype(int)
    X = expanded_df.drop(columns=[col for col in expanded_df.columns if target_column in col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation=activation)
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    _, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Model accuracy: {acc:.4f}")
    model.save(model_name)
    return model


if __name__ == "__main__":
    while True:
        try:
            # Option 1: Full pipeline
            run_full_pipeline = input("Would you like to run the preprocessing steps? (Y or N): ")
            if run_full_pipeline == 'Y':
                run_full_pipeline = True
            elif run_full_pipeline == 'exit':
                break
            else:
                run_full_pipeline = False
            
            if run_full_pipeline:
                dataset_path = fetch_dataset()
                preprocessed_data = preprocess_data(dataset_path)
                generated_rules = generate_rules(preprocessed_data)
                weighted_rules = compute_rule_weights(generated_rules)
                applied_rules_to_data = process_rules_for_records(weighted_rules, preprocessed_data)
                expanded_df = expand_normalized_rules(applied_rules_to_data)
                expanded_df.to_csv("rule_enhanced_dataset.csv", index=False)
            else:
                # Option 2: Reuse saved CSV
                expanded_df = pd.read_csv("rule_enhanced_dataset.csv")
            
            choice = input("Please select from the following options otherwise press enter:\n\n1. Load a pre-trained model\n2. Train a new model\n3. Run Standard Tests on Dataset\n\nEnter Selection: ")
            
            match choice:
                case "1":
                    model_name = input("Enter the file name of the model: ")
                    new_data = input("Enter the file path that contains the new records to predict: ")
                    preprocessed_new_data = preprocess_data(new_data)
                    model = load_model(model_name)
                    predictions = model.predict(preprocessed_new_data)
                    print(predictions)
                    break
                case "2":
                    model_name = input("Enter the file name of the model you would like to train: ")
                    import_data = input("Enter the file path that contains the data you're using to train: ")
                    preprocessed_imported_data = preprocess_data(import_data)
                    target_column = input("Enter the name of the target column relative to which the error will be calculated: ")
                    loss_method = int(input("Is the output classification binary or multi-class?\n\n1. Binary Classification\n2. Multi-Class Classification\n\nEnter your selection: "))
                    if loss_method == 1:
                        loss_method = "binary_crossentropy"
                        activation = "sigmoid"
                    elif loss_method == 2:
                        loss_method = "categorical_crossentropy"
                        activation = "softmax"
                    else:
                        print("Invalid selection")
                    max_epochs = int(input("Enter the max number of epochs for training: "))
                    model = train_nn(expanded_df=preprocessed_imported_data, model_name=model_name, target_column=target_column, loss=loss_method, activation=activation, epochs=max_epochs)
                    print("Model training complete")
                    break
                case "3":
                    print("\nRunning baseline experiment comparisons...\n")

                    # === 1. Rule-Augmented Baseline (Original + Rule Weights + Risk Score) ===
                    print("==> Training: Rule-Augmented Baseline")
                    baseline_cols = [c for c in expanded_df.columns if not c.startswith("Rule_")]
                    baseline_df = expanded_df.copy()
                    model = train_nn(
                        expanded_df=baseline_df,
                        model_name="baseline_rule_augmented.keras",
                        target_column="diagnosed_diabetes",
                        loss="binary_crossentropy",
                        activation="sigmoid",
                        epochs=20
                    )

                    # === 2. Rule Risk Score (Rule Weights + Risk Score only) ===
                    print("\n==> Training: Rule Risk Score Only")
                    rule_risk_cols = [c for c in expanded_df.columns if c.startswith("Rule_") or c == "Rule_Risk_Score"]
                    rule_risk_df = expanded_df[rule_risk_cols + ["diagnosed_diabetes"]]
                    model = train_nn(
                        expanded_df=rule_risk_df,
                        model_name="baseline_rule_risk_only.keras",
                        target_column="diagnosed_diabetes",
                        loss="binary_crossentropy",
                        activation="sigmoid",
                        epochs=20
                    )

                    # === 3. Normalized Rule Weights Only ===
                    print("\n==> Training: Normalized Rule Weights Only")
                    rule_only_cols = [c for c in expanded_df.columns if c.startswith("Rule_")]
                    rule_only_df = expanded_df[rule_only_cols + ["diagnosed_diabetes"]]
                    model = train_nn(
                        expanded_df=rule_only_df,
                        model_name="baseline_rule_weights_only.keras",
                        target_column="diagnosed_diabetes",
                        loss="binary_crossentropy",
                        activation="sigmoid",
                        epochs=20
                    )

                    print("\n*** Baseline experiments complete — models saved in current directory. ***\n")
                    break
                case "exit":
                    break
        except Exception as e:
            print(f"An error occurred: {e}")
            

    