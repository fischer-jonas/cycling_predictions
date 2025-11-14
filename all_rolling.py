import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, average_precision_score

# --- Load data ---
data = pd.read_csv("race_results_2017_2023.csv")
courses = pd.read_csv("structured_course_data.csv")

# --- Rank numeric, DNF = 1000 ---
data['Rank'] = pd.to_numeric(data['Rank'], errors='coerce').fillna(1000)

# --- Rolling-Features ---
data = data.sort_values(['Name', 'Date'])
window = 3
data['Top10Flag'] = (data['Rank'] <= 10).astype(int)

rolling_features = ['Rank', 'UCI points', 'PCS points', 'Top10Flag']
for col in rolling_features:
    data[f'Rolling_{col}'] = data.groupby('Name')[col].transform(
        lambda x: x.shift().rolling(window, min_periods=1).mean()
    )
    
# --- Plot Rolling-Features for Pogi and Wout---  
drivers = ["POGAČAR Tadej", "VAN AERT Wout"]
data['Date'] = pd.to_datetime(data['Date'], format='%d %B %Y')

plt.figure(figsize=(12, 6))

for d in drivers:
    df = data[data['Name'] == d].sort_values('Date')
    plt.scatter(df['Date'], df['Rolling_Rank'], s=10, label=d)

plt.ylim(200, 1)  
ax = plt.gca()
dates = data.sort_values('Date')['Date'].unique()
ax.set_xticks(dates[::100]) # Show only every 100th date on x-axis
plt.xticks(rotation=45)
plt.title("Rolling_Rank")
plt.xlabel("Date")
plt.ylabel("Rank")
plt.legend()
plt.tight_layout()
plt.show()

# --- Merge with Course data ---
merged = pd.merge(data, courses, on="Race Name", how="inner")

# --- Target label ---
merged['Win'] = (merged['Rank'] <= 10).astype(int)

# --- Feature-Selection ---
drop_cols = [
    'Win','Rank','Name','Race Name','Date','Time','Unnamed: 0',
    'Alpine','Team Time Trial','Access Road','Singletrack','Unknown','UCI points', 'PCS points', 'Top10Flag'
]
X = merged.drop(columns=drop_cols)
X = X.select_dtypes(include=[np.number])
y = merged['Win']

# --- Train-Test Split due to Races ---
train_races, test_races = train_test_split(merged['Race Name'].unique(), test_size=0.2, random_state=42)
X_train = X[merged['Race Name'].isin(train_races)]
X_test = X[merged['Race Name'].isin(test_races)]
y_train = y[merged['Race Name'].isin(train_races)]
y_test = y[merged['Race Name'].isin(test_races)]
names_test = merged.loc[merged['Race Name'].isin(test_races), ['Race Name', 'Name']]

# --- Scale pos weight ---
scale_pos_weight = (len(y_train)-sum(y_train))/sum(y_train)

# --- GridSearchCV Parameter ---
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

xgb = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss')

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='average_precision',  # PR-AUC
    cv=3,
    verbose=1,
    n_jobs=-1
)

# --- Training with GridSearch ---
grid.fit(X_train, y_train)

print("Beste Parameter:", grid.best_params_)

# --- Prediction ---
best_model = grid.best_estimator_
y_pred_prob = best_model.predict_proba(X_test)[:,1]
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
print("PR-AUC (Win):", average_precision_score(y_test, y_pred_prob))

# --- Winner for each Race ---
results = names_test.copy()
results['WinProb'] = y_pred_prob
results['WinProbNorm'] = results.groupby('Race Name')['WinProb'].transform(lambda x: x / x.sum())
predicted_winners = results.loc[results.groupby('Race Name')['WinProbNorm'].idxmax()]

print("\nVorhergesagte Sieger (Top10):")
print(predicted_winners[['Race Name', 'Name', 'WinProbNorm']].sort_values('WinProbNorm', ascending=False).head(10))

# --- Show Feature Importance ---
importances = pd.Series(best_model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance (Top10 Prediction)")
plt.show()
