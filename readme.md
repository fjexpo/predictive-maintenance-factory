# 🏭 Predictive Maintenance Analysis with Sensor Data

This project analyzes IoT sensor data from an industrial machine to predict potential machine failures. It uses real-world features like current, temperature, air quality, VOCs, and RPMs to build interpretable and accurate machine learning models.

## 📊 Dataset

- **Source:** Uploaded CSV
- **Target variable:** `fail` (1 = machine failure, 0 = normal)
- **Features include:**
  - `footfall`, `tempMode`, `AQ`, `USS`, `CS`, `VOC`, `RP`, `IP`, `Temperature`

## 🚀 Objectives

1. Clean and prepare the dataset.
2. Engineer useful features.
3. Train machine learning models (Random Forest, XGBoost).
4. Evaluate model performance.
5. Derive business insights and predictive signals.

## 🛠 Tech Stack

- Python, Pandas, Scikit-learn, XGBoost, Seaborn, Matplotlib

## 📌 Key Takeaways

- **XGBoost** achieved high recall in identifying potential machine failures.
- **Current Sensor (CS)** and **VOC levels** were strong indicators of failures.
- Predictive maintenance models can reduce downtime and optimize operations.

## 📁 Project Structure

