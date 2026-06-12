# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("car data.csv")

# Show first 5 rows
print(df.head())

# Dataset Information
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Drop Car Name column
df.drop(['Car_Name'], axis=1, inplace=True)

# Create new feature: Car Age
df['Current_Year'] = 2025
df['No_of_Years'] = df['Current_Year'] - df['Year']

# Drop unnecessary columns
df.drop(['Year', 'Current_Year'], axis=1, inplace=True)

# Convert categorical data into numerical
df = pd.get_dummies(df, drop_first=True)

# Split Features and Target
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Compare Actual vs Predicted
comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
})

print(comparison.head())

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.savefig("actual.png")
plt.show()