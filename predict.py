"""
Medical Insurance Cost Predictor
Enter your details and get an estimated insurance cost!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

# ─── Train the model ──────────────────────────────────────────────

print("=" * 55)
print("  Medical Insurance Cost Estimation")
print("  Training the prediction model...")
print("=" * 55)

# Load dataset
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insurance.csv")
df = pd.read_csv(csv_path, encoding='latin1')

# Store original charges stats for inverse-scaling later
charges_mean = df['charges'].mean()
charges_std = df['charges'].std()

# Prepare encoded data
categorical_columns = df.select_dtypes(include=['object', 'str']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)

# Standardize numerical features
scaler = StandardScaler()
df_encoded[['age', 'bmi', 'charges']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'charges']])

# Split and train
X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n  Model trained! (R² accuracy: {r2*100:.1f}%)\n")

# Store scaler params for age and bmi
age_mean = df['age'].mean()
age_std = df['age'].std()
bmi_mean = df['bmi'].mean()
bmi_std = df['bmi'].std()


# ─── Interactive Prediction ────────────────────────────────────────

def predict_cost():
    """Get user input and predict insurance cost."""
    print("-" * 55)
    print("  Enter your details below:")
    print("-" * 55)

    # Age
    while True:
        try:
            age = int(input("\n  Age (e.g., 25): "))
            if 0 < age < 120:
                break
            print("  Please enter a valid age (1-119)")
        except ValueError:
            print("  Please enter a number")

    # Sex
    while True:
        sex = input("  Sex (male/female): ").strip().lower()
        if sex in ['male', 'female']:
            break
        print("  Please enter 'male' or 'female'")

    # BMI
    while True:
        try:
            bmi = float(input("  BMI (e.g., 27.5): "))
            if 10 < bmi < 60:
                break
            print("  Please enter a valid BMI (10-60)")
        except ValueError:
            print("  Please enter a number")

    # Children
    while True:
        try:
            children = int(input("  Number of children (e.g., 2): "))
            if 0 <= children <= 10:
                break
            print("  Please enter a valid number (0-10)")
        except ValueError:
            print("  Please enter a number")

    # Smoker
    while True:
        smoker = input("  Smoker? (yes/no): ").strip().lower()
        if smoker in ['yes', 'no']:
            break
        print("  Please enter 'yes' or 'no'")

    # Region
    while True:
        region = input("  Region (northeast/northwest/southeast/southwest): ").strip().lower()
        if region in ['northeast', 'northwest', 'southeast', 'southwest']:
            break
        print("  Please enter one of: northeast, northwest, southeast, southwest")

    # ─── Build input for model ───
    # Standardize age and bmi
    age_scaled = (age - age_mean) / age_std
    bmi_scaled = (bmi - bmi_mean) / bmi_std

    # One-hot encode categorical inputs
    cat_input = pd.DataFrame([[sex, smoker, region]], columns=categorical_columns)
    cat_encoded = encoder.transform(cat_input)

    # Build feature vector (age, bmi, children, + one-hot columns)
    features = np.concatenate([[age_scaled, bmi_scaled, children], cat_encoded[0]])
    feature_names = ['age', 'bmi', 'children'] + list(encoder.get_feature_names_out(categorical_columns))
    input_df = pd.DataFrame([features], columns=feature_names)

    # Predict (standardized value)
    prediction_scaled = model.predict(input_df)[0]

    # Inverse-scale to get actual dollar amount
    prediction = prediction_scaled * charges_std + charges_mean

    # Display result
    print("\n" + "=" * 55)
    print(f"  Estimated Annual Insurance Cost: ${prediction:,.2f}")
    print("=" * 55)
    print(f"\n  Your Details:")
    print(f"    Age: {age} | Sex: {sex} | BMI: {bmi}")
    print(f"    Children: {children} | Smoker: {smoker} | Region: {region}")
    print()

    return prediction


# ─── Main Loop ─────────────────────────────────────────────────────

if __name__ == "__main__":
    while True:
        predict_cost()
        again = input("  Predict another? (yes/no): ").strip().lower()
        if again != 'yes':
            print("\n  Thank you! Goodbye.\n")
            break
