import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = r"C:/Users/HP/supply-chain-delay/data/interim/smart_logistics_cleaned.csv"
FIGURE_PATH = "notebooks/reports/figures/"

# Ensure the figures folder exists
os.makedirs(FIGURE_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded:", df.shape)
print("Columns in dataset:", df.columns.tolist())

# -------------------------------
# 1. Delay rate by traffic status
# -------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="traffic_status", y="logistics_delay", data=df)
plt.title("Delay rate by Traffic Status")
plt.savefig(os.path.join(FIGURE_PATH, "delay_by_traffic_status.png"))
plt.close()

# -------------------------------
# 2. Delay rate by delay reason
# -------------------------------
plt.figure(figsize=(10, 5))
sns.barplot(x="logistics_delay_reason", y="logistics_delay", data=df)
plt.title("Delay rate by Delay Reason")
plt.xticks(rotation=45)
plt.savefig(os.path.join(FIGURE_PATH, "delay_by_reason.png"))
plt.close()

# -------------------------------
# 3. Waiting time vs delay
# -------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x="logistics_delay", y="waiting_time", data=df)
plt.title("Waiting Time vs Logistics Delay")
plt.savefig(os.path.join(FIGURE_PATH, "waiting_time_vs_delay.png"))
plt.close()

# -------------------------------
# 4. Temperature vs delay
# -------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x="logistics_delay", y="temperature", data=df)
plt.title("Temperature vs Logistics Delay")
plt.savefig(os.path.join(FIGURE_PATH, "temperature_vs_delay.png"))
plt.close()

# -------------------------------
# 5. Humidity vs delay
# -------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(x="logistics_delay", y="humidity", data=df)
plt.title("Humidity vs Logistics Delay")
plt.savefig(os.path.join(FIGURE_PATH, "humidity_vs_delay.png"))
plt.close()

print(f"📊 Figures saved in {FIGURE_PATH}")
