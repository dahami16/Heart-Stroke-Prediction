import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy control variables
cigs_per_day = ctrl.Antecedent(np.arange(0, 21, 1), 'cigs_per_day')  # Range: 0 to 20
tot_chol = ctrl.Antecedent(np.arange(100, 301, 1), 'tot_chol')       # Range: 100 to 300
bmi = ctrl.Antecedent(np.arange(10, 41, 1), 'bmi')                  # Range: 10 to 40
age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')                  # Range: 20 to 80
sysBP = ctrl.Antecedent(np.arange(80, 201, 1), 'sysBP')             # Range: 80 to 200
susceptibility = ctrl.Consequent(np.arange(0, 101, 1), 'susceptibility')  # Range: 0 to 100

# Define fuzzy membership functions
for var, low, high in zip(
    [cigs_per_day, tot_chol, bmi, age, sysBP, susceptibility],
    [(0, 0, 10), (100, 150, 200), (10, 20, 30), (20, 40, 60), (80, 120, 160), (0, 25, 50)],
    [(10, 20, 20), (200, 250, 300), (30, 40, 40), (60, 70, 80), (160, 200, 200), (50, 75, 100)]
):
    var['low'] = fuzz.trimf(var.universe, low)
    var['high'] = fuzz.trimf(var.universe, high)

# Define rules
rules = [
    ctrl.Rule(cigs_per_day['low'] & tot_chol['low'] & bmi['low'] & age['low'] & sysBP['low'], susceptibility['low']),
    ctrl.Rule(cigs_per_day['high'] | tot_chol['high'] | bmi['high'] | age['high'] | sysBP['high'], susceptibility['high']),
]

# Create control system
susceptibility_ctrl = ctrl.ControlSystem(rules)
susceptibility_sim = ctrl.ControlSystemSimulation(susceptibility_ctrl)

# Streamlit app UI
st.title("Fuzzy Susceptibility Prediction")
st.write("Provide the input values:")

# User input fields
cigs_per_day_input = st.number_input("Cigarettes per Day:", min_value=0, max_value=20, value=0, step=1)
tot_chol_input = st.number_input("Total Cholesterol (mg/dL):", min_value=100, max_value=300, value=180, step=1)
bmi_input = st.number_input("BMI:", min_value=10.0, max_value=40.0, value=16.0, step=0.1)
age_input = st.number_input("Age:", min_value=20, max_value=80, value=24, step=1)
sysBP_input = st.number_input("Systolic Blood Pressure:", min_value=80, max_value=200, value=130, step=1)

# Prediction button
if st.button("Predict"):
    # Set fuzzy inputs
    susceptibility_sim.input['cigs_per_day'] = cigs_per_day_input
    susceptibility_sim.input['tot_chol'] = tot_chol_input
    susceptibility_sim.input['bmi'] = bmi_input
    susceptibility_sim.input['age'] = age_input
    susceptibility_sim.input['sysBP'] = sysBP_input

    # Compute fuzzy logic system
    susceptibility_sim.compute()

    # Display results
    st.write(f"Predicted Susceptibility: {susceptibility_sim.output['susceptibility']:.2f}")
