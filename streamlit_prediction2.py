import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

st.title("Chiller Load Estimation Using Decision Tree Regressor")

sensor_data = pd.read_csv('sensor_data.csv', encoding='latin1')
plan_efficiency_data = pd.read_csv('plant_ton_efficiency.csv')

merged_data = pd.merge(sensor_data, plan_efficiency_data, on='Time', how='outer')

merged_data.fillna(merged_data.mean(numeric_only=True), inplace=True)
merged_data['Time'] = pd.to_datetime(merged_data['Time'], errors='coerce')

X = merged_data[['GPM', 'DeltaCHW', 'WBT_C', 'RH [%]', 'kW_Tot']]
y = merged_data['CH Load']

corr_matrix = merged_data[['GPM', 'DeltaCHW', 'WBT_C', 'RH [%]', 'kW_Tot', 'CH Load']].corr()
st.write("Correlation matrix:")
st.write(corr_matrix['CH Load'].sort_values(ascending=False))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
st.write(f'Decision Tree MSE: {mse_tree}')
st.write(f'Decision Tree R-squared: {r2_tree}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tree, color='green', marker='x', label='Predicted')
plt.scatter(y_test, y_test, color='blue', marker='o', label='Actual', alpha=0.5)
plt.xlabel('Actual CH Load')
plt.ylabel('Predicted CH Load')
plt.title('Actual vs Predicted CH Load (Decision Tree)')
plt.legend()
st.pyplot(plt)

feature_importances = tree_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.title('Feature Importances for Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)

st.write("### Enter values for prediction:")
user_inputs = {}

for column in X_train.columns:
    min_val = X_train[column].min()
    max_val = X_train[column].max()
    user_input_value = st.number_input(f"Enter value for {column} (range {min_val:.2f} - {max_val:.2f})", min_value=float(min_val), max_value=float(max_val), format="%.2f")
    
    user_inputs[column] = user_input_value

user_input_df = pd.DataFrame([user_inputs])

if st.button("Calculate Predicted CH Load"):
    st.write("User Input DataFrame:")
    st.write(user_input_df)

    predicted_ch_load = tree_model.predict(user_input_df)
    st.write(f"### Predicted CH Load: {predicted_ch_load[0]:.2f}")
