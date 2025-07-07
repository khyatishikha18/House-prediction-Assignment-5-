import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save test Ids
test_ids = test["Id"]

# Separate target and combine data
y = train["SalePrice"]
train.drop("SalePrice", axis=1, inplace=True)
all_data = pd.concat([train, test], sort=False)

# Fill missing values
all_data.fillna(all_data.median(numeric_only=True), inplace=True)
all_data.fillna("None", inplace=True)

# Label encoding for object columns
le = LabelEncoder()
for col in all_data.select_dtypes(include="object").columns:
    all_data[col] = le.fit_transform(all_data[col].astype(str))

# Scale features
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_data)
X_train = all_scaled[:len(y)]
X_test = all_scaled[len(y):]

# Model
model = LinearRegression()
model.fit(X_train, y)
predictions = model.predict(X_test)

# 🔵 Visualization: Histogram of Predicted Prices
plt.figure(figsize=(10,6))
sns.histplot(predictions, kde=True, color='skyblue', bins=30)
plt.title("Distribution of Predicted House Prices", fontsize=16)
plt.xlabel("Predicted Price", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save prediction to CSV
submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
submission.to_csv("submission.csv", index=False)
