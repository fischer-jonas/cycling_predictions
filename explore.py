import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Load race results
# ------------------------------
data = pd.read_csv("race_results_2017_2023.csv")

# ------------------------------
# Convert Date to datetime and extract components
# ------------------------------
data['Date'] = pd.to_datetime(data['Date'], format='%d %B %Y')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.day_name()

print(data.head())

# ------------------------------
# Load course data
# ------------------------------
courses = pd.read_csv("structured_course_data.csv")

# ------------------------------
# Merge race results with course data
# ------------------------------
merged = pd.merge(data, courses, on="Race Name", how="inner")

# ------------------------------
# Count unique drivers
# ------------------------------
driver_column = "Name"
num_drivers = data[driver_column].nunique()
print(f"{num_drivers} different drivers")

# ------------------------------
# Cumulative UCI points per rider and per team
# ------------------------------
cum_uci_points = data.groupby('Name')['UCI points'].sum().sort_values(ascending=False)
cum_uci_points_teams = data.groupby('Team')['UCI points'].sum().sort_values(ascending=False)

# ------------------------------
# Find best rider per course feature
# ------------------------------
for col in courses.columns:
    if col not in ["Unnamed: 0", "Race Name"]:
        result = (
            merged[merged[col] > 0]
            .groupby('Name')['UCI points']
            .sum()
            .sort_values(ascending=False)
        )
        if not result.empty:
            print(f"Best rider on {col} is: {result.index[0]}")
