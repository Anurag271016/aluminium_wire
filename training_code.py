from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Load data
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# EDA Section
# ---- Cell Separator ----
print("Initial Data Overview:")
print("Shape:", df.shape)
df.head()

# ---- Cell Separator ----
# Histograms of raw data
df.hist(figsize=(12, 8), bins=30)
plt.suptitle("Raw Feature Distributions (Before Cleaning)", fontsize=16)
plt.tight_layout()
plt.show()

# ---- Cell Separator ----
# Boxplot of raw data
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Casting_Temperature_C', 'Rolling_Speed_m_min', 'Cooling_Rate_C_s',
                     'UTS_MPa', 'Elongation_%', 'Conductivity_%_IACS']])
plt.title("Box Plot of Features (Before Scaling)")
plt.xticks(rotation=45)
plt.show()

# ---- Cell Separator ----
# Data Info and Missing Values
print("\nData Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()

# ---- Cell Separator ----
# Define features
input_features = ['Casting_Temperature_C', 'Rolling_Speed_m_min', 'Cooling_Rate_C_s']
target_features = ['UTS_MPa', 'Elongation_%', 'Conductivity_%_IACS']

# Handle missing values
X = df[input_features].fillna(df[input_features].mean())
y = df[target_features].fillna(df[target_features].mean())

# CRITICAL FIX: Split data BEFORE scaling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply scaling ONLY to training data and transform others
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit ONLY on training data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---- Cell Separator ----
# Plot scaled input features (training data only)
scaled_train_df = pd.DataFrame(X_train_scaled, columns=input_features)
scaled_train_df.hist(figsize=(12, 6), bins=30)
plt.suptitle("Scaled Input Feature Distributions (Training Data)", fontsize=16)
plt.tight_layout()
plt.show()

# Boxplot of scaled input features
plt.figure(figsize=(10, 5))
sns.boxplot(data=scaled_train_df)
plt.title("Box Plot of Scaled Input Features (Training Data)")
plt.xticks(rotation=45)
plt.show()

# ---- Cell Separator ----
# Evaluation function
def evaluate(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

# Train models using CORRECTLY scaled data
models = {}
print("\nModel Training and Evaluation:")
for col in target_features:
    print(f"\nTraining model for {col}...")
    
    # Train Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_scaled, y_train[col])
    rf_preds = rf_model.predict(X_val_scaled)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train[col])
    xgb_preds = xgb_model.predict(X_val_scaled)
    
    # Compare performance
    rf_metrics = evaluate(y_val[col], rf_preds)
    xgb_metrics = evaluate(y_val[col], xgb_preds)
    
    print(f"Random Forest {col} Evaluation:", rf_metrics)
    print(f"XGBoost {col} Evaluation:", xgb_metrics)
    
    # Select best model based on R2 score
    if rf_metrics['R2'] > xgb_metrics['R2']:
        print(f"Selected Random Forest for {col}")
        models[col] = rf_model
    else:
        print(f"Selected XGBoost for {col}")
        models[col] = xgb_model

# ---- Cell Separator ----
# Parameter simulation with corrected scaling
print("\nRunning Parameter Simulation...")
casting_temp = np.linspace(X_train['Casting_Temperature_C'].min(), 
                           X_train['Casting_Temperature_C'].max(), 10)
rolling_speed = np.linspace(X_train['Rolling_Speed_m_min'].min(), 
                            X_train['Rolling_Speed_m_min'].max(), 10)
cooling_rate = np.linspace(X_train['Cooling_Rate_C_s'].min(), 
                           X_train['Cooling_Rate_C_s'].max(), 10)

combinations = list(product(casting_temp, rolling_speed, cooling_rate))
sim_df = pd.DataFrame(combinations, columns=input_features)

# Scale using training-based scaler
X_sim_scaled = scaler.transform(sim_df)

# Predict
sim_df['UTS_Pred'] = models['UTS_MPa'].predict(X_sim_scaled)
sim_df['Elongation_Pred'] = models['Elongation_%'].predict(X_sim_scaled)
sim_df['Conductivity_Pred'] = models['Conductivity_%_IACS'].predict(X_sim_scaled)

# Find optimal parameters
best = sim_df.sort_values(by=['UTS_Pred', 'Elongation_Pred', 'Conductivity_Pred'], 
                          ascending=False).head(1)
print("\n🔎 Optimal Parameter Settings:")
print(best)

# ---- Cell Separator ----
# What-if analysis: Cooling rate effect
print("\nRunning What-if Analysis...")
fixed_ct = X_train['Casting_Temperature_C'].mean()
fixed_rs = X_train['Rolling_Speed_m_min'].mean()
cooling_range = np.linspace(X_train['Cooling_Rate_C_s'].min(), 
                            X_train['Cooling_Rate_C_s'].max(), 30)

what_if_df = pd.DataFrame({
    'Casting_Temperature_C': fixed_ct,
    'Rolling_Speed_m_min': fixed_rs,
    'Cooling_Rate_C_s': cooling_range
})

# Scale and predict
X_whatif_scaled = scaler.transform(what_if_df)
what_if_df['UTS_Pred'] = models['UTS_MPa'].predict(X_whatif_scaled)
what_if_df['Elongation_Pred'] = models['Elongation_%'].predict(X_whatif_scaled)
what_if_df['Conductivity_Pred'] = models['Conductivity_%_IACS'].predict(X_whatif_scaled)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(cooling_range, what_if_df['UTS_Pred'], label='UTS')
plt.plot(cooling_range, what_if_df['Elongation_Pred'], label='Elongation')
plt.plot(cooling_range, what_if_df['Conductivity_Pred'], label='Conductivity')
plt.xlabel('Cooling Rate (°C/s)')
plt.ylabel('Predicted Value')
plt.title('What-if: Cooling Rate Effect')
plt.legend()
plt.grid(True)
plt.show()

# ---- Cell Separator ----
# Save models and scaler
print("\nSaving models and scaler...")
joblib.dump(models['UTS_MPa'], 'best_uts_model.joblib')
joblib.dump(models['Elongation_%'], 'best_elongation_model.joblib')
joblib.dump(models['Conductivity_%_IACS'], 'best_conductivity_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Save training ranges for Streamlit validation
training_ranges = {
    'Casting_Temperature_C': (X_train['Casting_Temperature_C'].min(), 
                              X_train['Casting_Temperature_C'].max()),
    'Rolling_Speed_m_min': (X_train['Rolling_Speed_m_min'].min(), 
                            X_train['Rolling_Speed_m_min'].max()),
    'Cooling_Rate_C_s': (X_train['Cooling_Rate_C_s'].min(), 
                         X_train['Cooling_Rate_C_s'].max())
}
joblib.dump(training_ranges, 'training_ranges.joblib')

print("\n✅ Training completed successfully! Models and scaler saved.")
