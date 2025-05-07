import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from matplotlib.ticker import MaxNLocator

# Load the dataset
df = pd.read_excel('Online Retail.xlsx')

# Drop rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Remove canceled orders (InvoiceNo starting with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Snapshot date
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# RFM calculation
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# Add Average Order Value (AOV)
avg_order_value = df.groupby('CustomerID')['TotalPrice'].sum() / df.groupby('CustomerID')['InvoiceNo'].nunique()
rfm['AOV'] = avg_order_value

# Simple CLV proxy
rfm['CLV'] = rfm['Monetary'] * rfm['Frequency'] / (rfm['Recency'] + 1)

# Features and target
X = rfm[['Recency', 'Frequency', 'Monetary', 'AOV']]
y = rfm['CLV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model training
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Predict full CLV
rfm['Predicted_CLV'] = model.predict(X)

# Segment customers based on Predicted CLV
rfm['Segment'] = pd.qcut(rfm['Predicted_CLV'], q=3, labels=['Low', 'Medium', 'High'])

# Visualizations
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', s=100, edgecolor='black')
plt.xlabel('Actual CLV', fontsize=14)
plt.ylabel('Predicted CLV', fontsize=14)
plt.title('Actual vs Predicted CLV', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('clv_scatter.png')

# Streamlit App
st.set_page_config(page_title="Customer Lifetime Value Prediction", layout="wide")
st.title("Customer Lifetime Value Prediction 📊")
st.markdown("""
    This app predicts customer lifetime value using RFM features and an XGBoost model.
    You can view the top predicted CLV customers and the full RFM/CLV data.
    """)

# Removed filter section

# Evaluation Section
st.subheader("Model Evaluation")
st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**MAE**: {mae:.2f}")
st.write(f"**R^2 Score**: {r2:.2f}")
st.image("clv_scatter.png", caption="Actual vs Predicted CLV", use_container_width=True)

# Display top customers based on predicted CLV
st.subheader("Top Customers by Predicted CLV")
top_customers = rfm.sort_values(by='Predicted_CLV', ascending=False).head(10)
st.dataframe(top_customers)

# Full RFM & CLV Table Section
st.subheader("Full RFM & CLV Table")
st.dataframe(rfm.reset_index())

# Export filtered data to Excel
if st.button("Export to Excel", key="export_button"):
    rfm.reset_index().to_excel("filtered_rfm_clv.xlsx", index=False)
    st.success("Filtered data exported successfully as 'filtered_rfm_clv.xlsx'")


