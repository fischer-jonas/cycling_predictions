# Cycling Race Prediction Using XGBoost

## Overview
This project predicts the likelihood of a rider finishing in the top 10 in professional cycling races using historical race results and structured course data.  
It is implemented in **Python**, using **XGBoost** as the main algorithm and includes **rolling features** to capture recent form.

The project contains four main scripts:

1. `explore.py` – general data exploration for all riders.  
2. `explore_wout.py` – exploration of races specifically for Wout Van Aert.  
3. `wout_machine_learning.py` – classification model to predict Wout wins using XGBoost.  
4. `all_rolling.py` – predictions for all riders using rolling averages as features.

---

## Features

### Rider Features
- Historical rank in previous races
- UCI points
- PCS points
- Rolling averages over last N races (form curves)

### Course Features
- Distance, elevation gain/loss
- Road type, surface type (paved, unpaved, cobblestones, etc.)

---

## Data
- `race_results_2017_2023.csv` – historical race results  
- `structured_course_data.csv` – structured course features per race  

---

## Target
- **Win** = top 10 finish (`1`)  
- **Rest** = below top 10 (`0`)  

---

## Models

### Wout Machine Learning (`wout_machine_learning.py`)
- **XGBoost Classifier** to predict if Wout Van Aert finishes top 10  
- Class imbalance handled with `scale_pos_weight`  
- GridSearchCV for hyperparameter tuning  

### All Riders Rolling Predictions (`all_rolling.py`)
- **XGBoost Classifier** for all riders using rolling features  
- Probabilities normalized per race to compare riders  
- Rolling averages capture recent performance trends  

---

## Performance (Example for All Riders)
| Metric       | Class 0 | Class 1 |
|--------------|--------|--------|
| Precision    | 0.94   | 0.16   |
| Recall       | 0.71   | 0.57   |
| F1-Score     | 0.81   | 0.25   |
| PR-AUC       | —      | 0.21   |

> Note: Class 1 (top 10) is highly imbalanced, so precision is low, but relative rankings are meaningful.

---

## Visualizations

### Feature Importance
Shows which numerical features contribute most to predictions.  
![Feature Importance](Plot/feature_importance.png)


### Rolling Features
Visualize recent performance trends per rider.  
![Rolling Features](Plot/rolling_features.png)

---

## Wout Van Aert Analysis

- `explore_wout.py` analyzes Wout-specific races and wins  
- Median race distance and preferred surface types for wins  
- Histograms and bar charts to visualize patterns in Wout’s victories  
![Wout Distance Histogram](Plot/wout_distance.png)
![Wout Surface Types](Plot/wout_surface.png)

---

## Example Usage

```python
# Fit model
model.fit(X_train, y_train)

# Predict probabilities
y_pred_prob = model.predict_proba(X_test)[:,1]

# Normalize probabilities per race
results['WinProbNorm'] = results.groupby('Race Name')['WinProb'].transform(lambda x: x / x.sum())

# Identify predicted winners
predicted_winners = results.loc[results.groupby('Race Name')['WinProbNorm'].idxmax()]
