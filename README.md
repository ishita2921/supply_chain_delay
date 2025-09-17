# 🚚 Supply Chain Delay Prediction
[![Run Tests](https://github.com/ishitamaithani/supply-chain-delay/actions/workflows/tests.yml/badge.svg)](https://github.com/ishitamaithani/supply-chain-delay/actions/workflows/tests.yml)

This project predicts **logistics and shipment delays** in a supply chain using historical smart logistics data.  
It applies **machine learning models** to classify whether a shipment will be delayed based on factors such as traffic, weather, waiting time, and demand forecast.  

---

## 📌 Project Overview
- **Goal:** Help companies anticipate shipment delays and optimize logistics decisions.  
- **Dataset:** Smart logistics dataset with features like timestamp, location, inventory, shipment status, temperature, humidity, traffic conditions, and demand forecast.  
- **Problem Type:** Binary classification (`Delayed` vs `Not Delayed`).  
- **Approach:** Data cleaning → Feature engineering → Model training → Evaluation → Deployment-ready artifacts.  

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **Database:** SQLite (for suppliers, locations, shipments management)  
- **Models Used:** Logistic Regression, Random Forest, XGBoost  
- **Tools:** Jupyter Notebook, VS Code  

---

## 📊 Results
- Best model: **Random Forest**  
- Test Accuracy: **91%**  
- ROC-AUC: **0.92**  
- Precision (delays): **1.00**  
- Recall (delays): **0.83**  

✔️ Balanced results, no overfitting (no more 100% accuracy issue).  
✔️ Useful for predicting logistics delays in real-world supply chain management.  

---

## 📈 Visualizations
- Delay rate by traffic status  
- Delay reasons distribution  
- Waiting time vs Delay  
- Environmental factors (temperature, humidity) vs Delay  

All figures are saved under: `notebooks/reports/figures/`

---

## 📂 Project Structure
