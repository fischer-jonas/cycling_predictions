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

# Display first rows
print(data.head())

# ------------------------------
# Filter data for Wout Van Aert
# ------------------------------
wout_data = data[data["Name"] == "VAN AERT Wout"]
print(wout_data.head())

# ------------------------------
# Load course data
# ------------------------------
courses = pd.read_csv("structured_course_data.csv")

# Merge Wout's race results with course data
courses_wout = pd.merge(wout_data, courses, on="Race Name", how="inner")

# ------------------------------
# Wout's wins
# ------------------------------
wout_wins = courses_wout[courses_wout["Rank"] == "1"]
print(f"Wout won {wout_wins.shape[0]} times")

# Compare feature medians for wins vs all participations
for col in courses_wout.columns:
    if np.issubdtype(courses_wout[col].dtype, np.number):
        if courses_wout[col].median() < wout_wins[col].median():
            print(f"{col} is preferred for wins")

# Median distance for wins
print(f"Median distance for a Wout win is {wout_wins['Distance'].median()} km")

# ------------------------------
# Histogram of distances
# ------------------------------
plt.hist(courses_wout['Distance'], bins=10, alpha=0.6, density=True, label="All participations")
plt.hist(wout_wins['Distance'], bins=10, alpha=0.6, density=True, label="Wins")
plt.xlabel("Distance (km)")
plt.ylabel("Density")
plt.title("Distribution of Race Distances for Wout Van Aert")
plt.legend()
plt.show()

# ------------------------------
# Bar plot of surface types
# ------------------------------
surface_cols = [
    'Street', 'Road', 'Paved', 'Asphalt', 'Path', 'Cycleway',
    'Unpaved', 'State Road', 'Cobblestones', 'Compacted Gravel',
    'Off-grid (unknown)', 'Singletrack', 'Access Road'
]

plt.figure(figsize=(10,6))
plt.bar(surface_cols, courses_wout[surface_cols].sum())
plt.xticks(rotation=45)
plt.xlabel("Surface Type")
plt.ylabel("Total Distance (km)")
plt.title("Surface Type Distribution for Wout Van Aert Races")
plt.tight_layout()
plt.show()
