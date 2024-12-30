# -*- coding: utf-8 -*-
"""Fuzzy System Implementation with Decision Tree Integration"""

import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def install_dependencies():
    """Ensure required libraries are installed."""
    try:
        import skfuzzy
    except ImportError:
        os.system('pip install scikit-fuzzy==0.4.2')


def plot_membership(x, memberships, labels, title):
    """Plot membership functions."""
    plt.figure(figsize=(10, 6))
    for membership, label in zip(memberships, labels):
        plt.plot(x, membership, label=label)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Membership Degree')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_membership_functions():
    """Define and plot fuzzy membership functions for various attributes."""
    # Define ranges
    age = np.arange(32, 71, 1)
    cigs = np.arange(0, 71, 1)
    sysBP = np.arange(83.5, 296, 1)
    cholesterol = np.arange(107, 697, 1)
    bmi = np.arange(15.96, 57, 1)

    # Define membership functions for each attribute
    age_memberships = [
        fuzz.trimf(age, [32, 32, 45]),
        fuzz.trimf(age, [40, 45, 55]),
        fuzz.trimf(age, [50, 55, 70]),
        fuzz.trimf(age, [65, 70, 70])
    ]
    cigs_memberships = [
        fuzz.trimf(cigs, [0, 0, 10]),
        fuzz.trimf(cigs, [11, 20, 30]),
        fuzz.trimf(cigs, [21, 50, 70])
    ]
    sysBP_memberships = [
        fuzz.trimf(sysBP, [83.5, 83.5, 120]),
        fuzz.trimf(sysBP, [121, 140, 160]),
        fuzz.trimf(sysBP, [141, 180, 295]),
        fuzz.trimf(sysBP, [181, 200, 295])
    ]
    cholesterol_memberships = [
        fuzz.trimf(cholesterol, [107, 107, 200]),
        fuzz.trimf(cholesterol, [200, 220, 239]),
        fuzz.trimf(cholesterol, [240, 696, 696])
    ]
    bmi_memberships = [
        fuzz.trimf(bmi, [15.96, 15.96, 25]),
        fuzz.trimf(bmi, [25.1, 30, 35]),
        fuzz.trimf(bmi, [30.1, 35, 56]),
        fuzz.trimf(bmi, [40, 45, 56])
    ]

    # Plot the membership functions
    plot_membership(age, age_memberships, ['Low', 'Moderate', 'High', 'Extremely High'], 'Age Membership Functions')
    plot_membership(cigs, cigs_memberships, ['Low', 'Moderate', 'High'], 'Cigarettes Per Day Membership Functions')
    plot_membership(sysBP, sysBP_memberships, ['Low', 'Moderate', 'High', 'Extremely High'],
                    'Systolic Blood Pressure Membership Functions')
    plot_membership(cholesterol, cholesterol_memberships, ['Low', 'Moderate', 'High'],
                    'Total Cholesterol Membership Functions')
    plot_membership(bmi, bmi_memberships, ['Low', 'Moderate', 'High', 'Extremely High'], 'BMI Membership Functions')


def preprocess_data(file_path):
    """Preprocess the dataset: load, clean, and impute missing values."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {file_path}")

    data = pd.read_csv(file_path)

    # Handle missing values
    columns_to_impute = ['cigsPerDay', 'totChol', 'BMI', 'age', 'sysBP']
    impute_values = {
        'cigsPerDay': data['cigsPerDay'].mean(),
        'totChol': data['totChol'].median(),
        'BMI': data['BMI'].mean(),
        'age': data['age'].mean(),
        'sysBP': data['sysBP'].mean()
    }

    for column in columns_to_impute:
        data[column].fillna(impute_values[column], inplace=True)

    print(f"Data loaded with {data.shape[0]} records after handling missing values.")
    return data


def create_fuzzy_system():
    """Create the fuzzy control system for susceptibility prediction."""
    # Define fuzzy variables
    cigs_per_day = ctrl.Antecedent(np.arange(0, 101, 1), 'cigsPerDay')
    tot_chol = ctrl.Antecedent(np.arange(107, 301, 1), 'totChol')
    bmi = ctrl.Antecedent(np.arange(15.96, 57, 1), 'BMI')
    age = ctrl.Antecedent(np.arange(32, 71, 1), 'age')
    sysBP = ctrl.Antecedent(np.arange(83.5, 296, 1), 'sysBP')
    susceptibility = ctrl.Consequent(np.arange(0, 101, 1), 'susceptibility')

    # Define fuzzy membership functions
    cigs_per_day['low'] = fuzz.trimf(cigs_per_day.universe, [0, 0, 10])
    cigs_per_day['moderate'] = fuzz.trimf(cigs_per_day.universe, [11, 20, 30])
    cigs_per_day['high'] = fuzz.trimf(cigs_per_day.universe, [21, 50, 70])

    tot_chol['low'] = fuzz.trimf(tot_chol.universe, [107, 107, 200])
    tot_chol['moderate'] = fuzz.trimf(tot_chol.universe, [200, 220, 239])
    tot_chol['high'] = fuzz.trimf(tot_chol.universe, [240, 300, 300])

    bmi['low'] = fuzz.trimf(bmi.universe, [15.96, 15.96, 25])
    bmi['moderate'] = fuzz.trimf(bmi.universe, [25.1, 30, 35])
    bmi['high'] = fuzz.trimf(bmi.universe, [30.1, 35, 56])
    bmi['extremely_high'] = fuzz.trimf(bmi.universe, [40, 45, 56])

    age['low'] = fuzz.trimf(age.universe, [32, 32, 45])
    age['moderate'] = fuzz.trimf(age.universe, [40, 45, 55])
    age['high'] = fuzz.trimf(age.universe, [50, 55, 70])
    age['extremely_high'] = fuzz.trimf(age.universe, [65, 70, 70])

    sysBP['low'] = fuzz.trimf(sysBP.universe, [83.5, 83.5, 120])
    sysBP['moderate'] = fuzz.trimf(sysBP.universe, [121, 140, 160])
    sysBP['high'] = fuzz.trimf(sysBP.universe, [141, 180, 295])
    sysBP['extremely_high'] = fuzz.trimf(sysBP.universe, [181, 200, 295])

    susceptibility['low'] = fuzz.trimf(susceptibility.universe, [0, 0, 50])
    susceptibility['moderate'] = fuzz.trimf(susceptibility.universe, [25, 50, 75])
    susceptibility['high'] = fuzz.trimf(susceptibility.universe, [50, 100, 100])

    return cigs_per_day, tot_chol, bmi, age, sysBP, susceptibility


def create_decision_tree_model(data):
    """Train a Decision Tree model on the dataset."""
    features = ['age', 'cigsPerDay', 'totChol', 'BMI', 'sysBP']
    target = 'TenYearCHD'

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    data['Decision_Tree_Probability'] = decision_tree_model.predict_proba(X)[:, 1]
    return decision_tree_model


def main():
    """Main function to execute the fuzzy system and decision tree model."""
    install_dependencies()

    # File paths
    dataset_path = r"/content/fuzzy risk factors dataset.csv"
    test_path = r"/content/testing csv.csv"

    # Step 1: Preprocess the data
    data = preprocess_data(dataset_path)

    # Step 2: Plot membership functions
    plot_membership_functions()

    # Step 3: Create fuzzy system
    cigs_per_day, tot_chol, bmi, age, sysBP, susceptibility = create_fuzzy_system()

    # Step 4: Train the Decision Tree model
    decision_tree_model = create_decision_tree_model(data)

    # Step 5: Save results to an Excel file
    data.to_excel(test_path, index=False)


if __name__ == "__main__":
    main()
