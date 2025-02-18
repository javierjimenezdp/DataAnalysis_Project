"""
# Data Analysis Project Documentation

## Objective
To identify anomalies in product quantities using the Isolation Forest anomaly detection model. Additionally, initial exploratory data analysis and handling of null values will be performed.

## Step by Step

### Step 1: Data Loading
```python
import pandas as pd

# Load data from a CSV file
df = pd.read_csv('path_to_file.csv')
```
**Description:**
- The pandas library is imported, which is used for data manipulation and analysis.
- A CSV file is loaded into a DataFrame `df`.

### Step 2: Handling Null Values
```python
# Show the count of null values per column
print(df.isnull().sum())
```
**Description:**
- `isnull()` returns a boolean DataFrame indicating if values are null.
- `sum()` is used to count the null values per column.

### Step 3: Imputing Null Values
```python
# Impute null values in the 'CustomerID' column with -1
df['CustomerID'].fillna(-1, inplace=True)
```
**Description:**
- `fillna(-1)` replaces null values in the `CustomerID` column with -1 to represent unknown customers.
- `inplace=True` modifies the original DataFrame.

### Step 4: Analyzing Quantities by StockCode
```python
# Group by 'StockCode' and sum quantities
df_grouped = df.groupby('StockCode')['Quantity'].sum().reset_index()
print(df_grouped)
```
**Description:**
- `groupby('StockCode')` groups the data by product code.
- `sum()` calculates the total quantity for each code.
- `reset_index()` returns the index to its original shape.

### Step 5: Anomaly Detection with Isolation Forest
```python
from sklearn.ensemble import IsolationForest

# Initialize and fit the Isolation Forest model
model = IsolationForest(contamination=0.05)
df['anomaly'] = model.fit_predict(df[['Quantity']])
```
**Description:**
- `IsolationForest` is an algorithm used to detect anomalies. `contamination` indicates the expected proportion of anomalies.
- `fit_predict()` fits the model to the data and predicts anomalies.

### Step 6: Filtering Anomalies
```python
# Filter anomalies
anomalies = df[df['anomaly'] == -1]
print(anomalies)
```
**Description:**
- Rows where `anomaly` equals -1 are filtered, indicating anomalies.

### Analysis Examples

#### Example 1: Relationship between Dates and Quantities
```python
# Group by 'InvoiceDate' and sum quantities
df_date_quantity = df.groupby('InvoiceDate')['Quantity'].sum().reset_index()
print(df_date_quantity)
```
**Description:**
- Quantities are grouped by invoice date to analyze daily sales volume.

#### Example 2: Relationship between Users and Sales Volumes
```python
# Group by 'CustomerID' and sum quantities
df_user_sales = df.groupby('CustomerID')['Quantity'].sum().reset_index()
print(df_user_sales)
```
**Description:**
- Quantities are grouped by customer ID to understand purchase volumes per customer.

#### Example 3: Anomaly Detection in Quantities
```python
# Anomaly detection visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(df['InvoiceDate'], df['Quantity'], c=df['anomaly'], cmap='coolwarm')
plt.title('Anomaly Detection in Quantities')
plt.xlabel('Invoice Date')
plt.ylabel('Quantity')
plt.show()
```
**Description:**
- Matplotlib is used to visualize quantities over time, where anomaly points are highlighted in a different color.
"""
