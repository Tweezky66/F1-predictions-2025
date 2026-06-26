import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


fastf1.Cache.enable_cache("f1_cache") #Caching the data

#It will alows us to map drivers capabilyties during rain weather,if it rains of course
RAIN_SKILL_INDEX = {
    "VER": 1.15, "HAM": 1.15, "ALO": 1.12, "NOR": 1.10, "LEC": 1.08,
    "RUS": 1.08, "SAI": 1.05, "PIA": 1.05, "GAS": 1.05, "OCO": 1.05,
    "STR": 1.02, "TSU": 1.00, "ALB": 1.00, "BOT": 1.00, "MAG": 0.95,
    "HUL": 0.95, "ZHO": 0.90, "SAR": 0.85, "PER": 0.95, "RIC": 1.00,
    "BEA": 1.00, "LAW": 1.00
}

#Scale 1-10 how hard is it to overtake,10 = Impossible to pass (Monaco), 1 = Easy to pass (Spa/Bahrain) depending on track architection
TRACK_OVERTAKING_DIFFICULTY = {
    "Monaco Grand Prix": 10.0, "Singapore Grand Prix": 9.0, 
    "Emilia Romagna Grand Prix": 8.0, "Dutch Grand Prix": 8.0, 
    "Spanish Grand Prix": 7.0, "Japanese Grand Prix": 7.0, 
    "Australian Grand Prix": 6.0, "Canadian Grand Prix": 5.0, 
    "Miami Grand Prix": 5.0, "British Grand Prix": 4.0, 
    "Italian Grand Prix": 4.0, "United States Grand Prix": 3.0, 
    "Mexico City Grand Prix": 3.0, "Azerbaijan Grand Prix": 3.0, 
    "São Paulo Grand Prix": 2.0, "Bahrain Grand Prix": 2.0, 
    "Belgian Grand Prix": 2.0, "Las Vegas Grand Prix": 2.0, 
    "Qatar Grand Prix": 4.0, "Abu Dhabi Grand Prix": 5.0, 
    "Saudi Arabian Grand Prix": 4.0, "Chinese Grand Prix": 4.0
}

#Different cars has differnt power uniqness,some better take turns,some not,  3 = High downforce (curvy/street), 2 = Medium (balanced), 1 = Low (power/straights)
TRACK_DOWNFORCE_LEVEL = {
    "Monaco Grand Prix": 3, "Singapore Grand Prix": 3, "Hungarian Grand Prix": 3, 
    "Dutch Grand Prix": 3, "Spanish Grand Prix": 3, "Japanese Grand Prix": 2, 
    "British Grand Prix": 2, "United States Grand Prix": 2, "São Paulo Grand Prix": 2, 
    "Bahrain Grand Prix": 2, "Miami Grand Prix": 2, "Qatar Grand Prix": 2, 
    "Abu Dhabi Grand Prix": 2, "Australian Grand Prix": 2, "Chinese Grand Prix": 2,
    "Italian Grand Prix": 1, "Belgian Grand Prix": 1, "Las Vegas Grand Prix": 1, 
    "Azerbaijan Grand Prix": 1, "Saudi Arabian Grand Prix": 1, 
    "Canadian Grand Prix": 1, "Mexico City Grand Prix": 2, "Emilia Romagna Grand Prix": 2
}

if not os.path.exists("f1_combined_dataset.csv"):
    years_to_load = [2025]
    masters_dataset = []

    for year in years_to_load:           #get both qualiying and race
        schedule = fastf1.get_event_schedule(year)
        total_rounds = schedule["RoundNumber"].max() 
        for round_num in range(1, total_rounds + 1):
            try:
                print(f"Proccesing {year} , Round {round_num}")

                quali = fastf1.get_session(year, round_num, "Q")
                quali.load(telemetry=False , weather=True, messages=False)
                grid_data = quali.results[['Abbreviation', 'Position']].rename(columns={"Position": "GridPosition"})

                race = fastf1.get_session(year , round_num, "R")
                race.load(telemetry=False, weather=True, messages=False)

                weather_data = race.weather_data
                is_wet = weather_data["Rainfall"].any() if not weather_data.empty else False

                track_event_name = race.event["EventName"]

                race_data = race.results[['Abbreviation', 'TeamName', 'ClassifiedPosition', 'Points', 'Status']]
            
                clean_laps = race.laps[race.laps["IsAccurate"] == True].copy()

                drivers_pace = clean_laps.groupby("Driver")["LapTime"].median()
                fastest_pace = drivers_pace.min()
                pace_delta = drivers_pace - fastest_pace

                pace_features = pd.DataFrame({
                "Abbreviation" : pace_delta.index,
                "PaceDelta (s)" : pace_delta.dt.total_seconds()
                })

                for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                    clean_laps[f"{col} (s)"] = clean_laps[col].dt.total_seconds()

            
                sector_features = clean_laps.groupby("Driver")[
                    ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
                ].mean().reset_index().rename(columns={"Driver" : "Abbreviation"})

            
            
            
                round_merged = pd.merge(grid_data , race_data, on = "Abbreviation", how="inner")
                round_merged = pd.merge(round_merged , pace_features , on="Abbreviation", how="left")
                round_merged = pd.merge(round_merged, sector_features, on="Abbreviation", how="left")

                round_merged["Year"] = year
                round_merged["Round"] = round_num
                round_merged["EventName"] = track_event_name
                round_merged["IsWetRace"] = int(is_wet)

                masters_dataset.append(round_merged)

            except Exception as e:
                print(f"Something went wrong, Skipping {year}, Round {round_num}, Error {e}")

    final_df = pd.concat(masters_dataset, ignore_index=True)
    final_df["ClassifiedPosition"] = pd.to_numeric(final_df["ClassifiedPosition"], errors="coerce").fillna(20)

    final_df["RainSkill"] = final_df["Abbreviation"].map(RAIN_SKILL_INDEX).fillna(1.0)
    final_df["Adjusted_PaceDelta"] = np.where(
        final_df["IsWetRace"] == 1,
        final_df["PaceDelta (s)"] / final_df["RainSkill"],
        final_df["PaceDelta (s)"]
    )

    final_df["IsPodium"] = np.where(final_df["ClassifiedPosition"] <= 3, 1, 0).astype(int)
    final_df["Overtaking_Difficulty"] = final_df["EventName"].map(TRACK_OVERTAKING_DIFFICULTY).fillna(5) 
    final_df["Track_Downforce"] = final_df["EventName"].map(TRACK_DOWNFORCE_LEVEL).fillna(2)

    #High numbers = heavily penalized for bad qualifying cause it will give a bad starter position
    final_df["Grid_Penalty_Score"] = final_df["GridPosition"] * final_df["Overtaking_Difficulty"]
    final_df = final_df.sort_values(by=["Year", "Round", "Abbreviation"])

    #We use shift(1) so the model doesn't use the current race to predict the current race
    final_df["Podiums_Last_3_Races"] = final_df.groupby("Abbreviation")["IsPodium"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0)

    final_df["HasStreak"] = np.where(final_df["Podiums_Last_3_Races"] >= 2, 1, 0).astype(int)
    final_df.to_csv("f1_combined_dataset.csv", index=False)
    print("Data Saved")
else:
    print("Dataset already exists! Skipping extraction step...")


#Define the model and target
print("Loading csv...")
df = pd.read_csv("f1_combined_dataset.csv")

latest_year = df["Year"].max()
latest_round = df[df["Year"] == latest_year]["Round"].max()

train_df = df[~((df["Year"] == latest_year) & (df["Round"] == latest_round))].copy()


test_df = df[(df["Year"] == latest_year) & (df["Round"] == latest_round)].copy()

features = [
    "GridPosition",  
    "Adjusted_PaceDelta", 
    "Track_Downforce", 
    "Grid_Penalty_Score", 
    "HasStreak", 
    "IsWetRace",
    "Podiums_Last_3_Races"
]

X_train_raw = train_df[features]
y_train = train_df["ClassifiedPosition"]

X_test_raw = test_df[features]
y_test = test_df["ClassifiedPosition"]



imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)


model = XGBRegressor(
n_estimators = 100,
learning_rate = 0.04,
max_depth = 5,
random_state = 42
)

model.fit(X_train, y_train)
raw_predictions = model.predict(X_test)

test_df["Raw_Predicted_Score"] = raw_predictions

test_df["Final_Predicted_Position"] = test_df["Raw_Predicted_Score"].rank(method='min').astype(int)

mae = mean_absolute_error(test_df["ClassifiedPosition"], test_df["Final_Predicted_Position"])

print(f"(MAE): {mae:.2f} positions")
print("(An MAE of 2.0 means the model was off by 2 grid slots on average)")

final_results = test_df.sort_values("Final_Predicted_Position").reset_index(drop=True)

podium = final_results.loc[:2]

print(f"\n🏆 Predicted Podium for {latest_year} Round {latest_round} 🏆")
print(f"🥇 P1: {podium.iloc[0]['Abbreviation']} (Predicted Pos: 1)")
print(f"🥈 P2: {podium.iloc[1]['Abbreviation']} (Predicted Pos: 2)")
print(f"🥉 P3: {podium.iloc[2]['Abbreviation']} (Predicted Pos: 3)")

# Visualize the results
plt.figure(figsize=(10,6))

errors = np.abs(test_df["ClassifiedPosition"] - test_df["Final_Predicted_Position"])

scatter = plt.scatter(test_df["ClassifiedPosition"], test_df["Final_Predicted_Position"], 
                      alpha=0.9, c=errors, cmap="viridis", edgecolors="black", s=100)

#Draw a red dashed line representing "Perfect Accuracy" (y = x)
plt.plot([1, 20], [1, 20], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.colorbar(scatter, label='Prediction Error (Positions Off)')

plt.title(f"Actual vs Predicted: {latest_year} Round {latest_round}", fontsize=14, fontweight="bold")
plt.xlabel("Actual Finishing Position", fontsize=12)
plt.ylabel("Predicted Finishing Position (Ranked)", fontsize=12)
plt.xticks(np.arange(1, 21, 1))
plt.yticks(np.arange(1, 21, 1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("results/predicted_vs_actual_r25.png")
plt.show()
