import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, average_precision_score

# ------------------------------
# Load data
# ------------------------------
data = pd.read_csv("race_results_2017_2023.csv")
courses = pd.read_csv("structured_course_data.csv")

# ------------------------------
# Convert rank to numeric
# DNF (Did Not Finish) -> assign a high number (e.g., 1000)
# ------------------------------
data['Rank'] = pd.to_numeric(data['Rank'], errors='coerce').fillna(1000)

# ------------------------------
# Compute rolling features to capture recent form
# Sort by rider and date first
# ------------------------------
data = data.sort_values(['Name', 'Date'])
window = 3
data['Top10Flag'] = (data['Rank'] <= 10).astype(int)

rolling_features = ['Rank', 'UCI points', 'PCS points', 'Top10Flag']
for col in rolling_features:
    data[f'Rolling_{col}'] = data.groupby('Name')[col].transform(
        lambda x: x.shift().rolling(window, min_periods=1).mean()
    )

# ------------------------------
# Merge race results with course features
# ------------------------------
merged = pd.merge(data, courses, on="Race Name", how="inner")

# ------------------------------
# Define target variable
# 1 if top 10 finish, 0 otherwise
# ------------------------------
merged['Win'] = (merged['Rank'] <= 10).astype(int)

# ------------------------------
# Select features: only numeric
# Drop columns that are not features
# ------------------------------
drop_cols = [
    'Win','Rank','Name','Race Name','Date','Time','Unnamed: 0',
    'Alpine','Team Time Trial','Access Road','Singletrack','Unknown'
]
X = merged.drop(columns=drop_cols)
X = X.select_dtypes(include=[np.number])
y = merged['Win']

# ------------------------------
# Split train/test by race to avoid data leakage
# ------------------------------
train_races, test_races = train_test_split(
    merged['Race Name'].unique(), test_size=0.2, random_state=42
)

X_train = X[merged['Race Name'].isin(train_races)]
X_test = X[merged['Race Name'].isin(test_races)]
y_train = y[merged['Race Name'].isin(train_races)]
y_test = y[merged['Race Name'].isin(test_races)]

# Keep rider names and races for later
names_test = merged.loc[merged['Race Name'].isin(test_races), ['Race Name', 'Name']]

# ------------------------------
# Handle class imbalance
# ------------------------------
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ------------------------------
# Define parameter grid for GridSearchCV
# ------------------------------
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

xgb = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='average_precision',  # PR-AUC for imbalanced dataset
    cv=3,
    verbose=2,  # show progress
    n_jobs=-1
)

# ------------------------------
# Fit model with GridSearch
# ------------------------------
grid.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid.best_params_)

# ------------------------------
# Make predictions on test set
# ------------------------------
best_model = grid.best_estimator_
y_pred_prob = best_model.predict_proba(X_test)[:,1]
y_pred = best_model.predict(X_test)

# ------------------------------
# Evaluation
# ------------------------------
print(classification_report(y_test, y_pred))
print("PR-AUC (Win):", average_precision_score(y_test, y_pred_prob))

# ------------------------------
# Assign probabilities to riders and normalize per race
# ------------------------------
results = names_test.copy()
results['WinProb'] = y_pred_prob

# Normalize probabilities per race
results['WinProbNorm'] = results.groupby('Race Name')['WinProb'].transform(lambda x: x / x.sum())

# ------------------------------
# Determine predicted winner per race
# ------------------------------
predicted_winners = results.loc[results.groupby('Race Name')['WinProbNorm'].idxmax()]

print("\nTop 10 predicted winners across all races:")
print(predicted_winners[['Race Name', 'Name', 'WinProbNorm']].sort_values('WinProbNorm', ascending=False).head(10))
