import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Optional: handle class imbalance
from imblearn.over_sampling import SMOTE

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv("race_results_2017_2023.csv")
courses = pd.read_csv("structured_course_data.csv")

# ------------------------------
# Filter data for a single rider (Wout Van Aert)
# ------------------------------
wout_data = data[data["Name"] == "VAN AERT Wout"]
courses_wout = pd.merge(wout_data, courses, on="Race Name", how="inner")

# ------------------------------
# Normalize surface type columns by race distance
# ------------------------------
surface_cols = [
    'Street', 'Road', 'Paved', 'Asphalt', 'Path', 'Cycleway',
    'Unpaved', 'State Road', 'Cobblestones', 'Compacted Gravel',
    'Off-grid (unknown)', 'Singletrack', 'Access Road'
]

for col in surface_cols:
    courses_wout[col] = courses_wout[col] / courses_wout['Distance']

# Replace NaN with 0 if distance was 0
courses_wout[surface_cols] = courses_wout[surface_cols].fillna(0)

# ------------------------------
# Define target variable
# 0 = top 10 finish, 1 = rest
# ------------------------------
courses_wout['Rank'] = pd.to_numeric(courses_wout['Rank'], errors='coerce').fillna(200)
courses_wout['RankClass'] = np.where(courses_wout['Rank'] < 10, 0, 1)

# ------------------------------
# Prepare features and target
# Drop non-numeric and irrelevant columns
# ------------------------------
X = courses_wout.drop(columns=[
    'RankClass','Rank','Name','Race Name','Team','Date','Time',
    'UCI points','PCS points','TimeAfterTeamTTT','Unnamed: 0',
    'Alpine','Team Time Trial','Unknown'
])
X = X.fillna(0)

y = courses_wout['RankClass']

# ------------------------------
# Split into train and test sets (stratified)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Scale features
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# XGBoost model and hyperparameter grid
# ------------------------------
model = XGBClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_weighted',  # better for imbalanced classes
    cv=3,
    verbose=1,
    n_jobs=-1
)

# ------------------------------
# Fit GridSearchCV
# ------------------------------
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and F1 score
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

# Use the best estimator
model = grid_search.best_estimator_

# ------------------------------
# Predict on test set
# ------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ------------------------------
# Evaluate performance
# ------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------
# Feature importance visualization
# ------------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8,6))
importances.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.show()
