import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'IMDb Movies India.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find 'IMDb Movies India.csv' in {script_dir}")

df = pd.read_csv(csv_path, encoding='latin-1')


df = df[df['Name'].str.strip() != ''].copy()

df['Year'] = df['Year'].str.extract('(\\d{4})').astype(float)

df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

df.dropna(subset=['Rating'], inplace=True)

df['Duration'] = df['Duration'].fillna(df['Duration'].median())

for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = df[col].fillna('Unknown')

df['Year'] = df['Year'].astype(int)

df['Genre_list'] = df['Genre'].str.split(', ').fillna('')
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df['Genre_list']), columns=mlb.classes_, index=df.index)
df = pd.concat([df, genre_encoded], axis=1)

df.drop(['Genre', 'Genre_list'], axis=1, inplace=True)

categorical_cols = ['Director', 'Actor 1', 'Actor 2', 'Actor 3']

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = ohe.fit_transform(df[categorical_cols])
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

df.drop('Name', axis=1, inplace=True)

X = df.drop('Rating', axis=1)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("\nTraining RandomForestRegressor model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")

y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.4f}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='black', alpha=0.6)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
plt.title('Actual vs. Predicted Movie Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual Ratings', fill=True)
sns.kdeplot(y_pred, label='Predicted Ratings', fill=True)
plt.title('Distribution of Actual and Predicted Movie Ratings')
plt.xlabel('Ratings')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()