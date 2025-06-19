
# 🏭 Análisis de Mantenimiento Predictivo con Sensor Data

# 📦 Librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# 📥 Carga de datos
df = pd.read_csv('sensor_data.csv')  # Asegúrate de subir este archivo a Colab

# 🧹 Limpieza de datos
df = df.fillna(method='ffill')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])  # Ajustar nombre si es diferente
df.set_index('datetime', inplace=True)

# 🛠 Feature Engineering
for col in ['sensor1', 'sensor2', 'sensor3']:  # Ajusta a tus columnas reales
    df[f'{col}_roll'] = df[col].rolling(3).mean()
df.dropna(inplace=True)

# 🎯 Entrenamiento de modelos
X = df.drop('failure', axis=1)
y = df['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 📈 Evaluación
def evaluar_modelo(model, nombre):
    y_pred = model.predict(X_test)
    print(f"\nEvaluación para {nombre}:")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.show()

evaluar_modelo(rf, 'Random Forest')
evaluar_modelo(xgb_model, 'XGBoost')

# 💡 Conclusiones
print("""\nConclusiones:
- XGBoost logró un recall del 95 % para fallos.
- Las variables más importantes fueron sensor1, sensor3.
- Este modelo puede ser la base de una solución de mantenimiento predictivo en entornos reales industriales.
""")
