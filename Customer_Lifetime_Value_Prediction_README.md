
# Customer Lifetime Value (CLV) Prediction App

## Overview
This app predicts Customer Lifetime Value (CLV) based on customer purchase behavior using RFM features (Recency, Frequency, Monetary) and an XGBoost regression model. The goal is to predict the lifetime value of customers to aid in targeted marketing efforts.

## Objective
- Predict the lifetime value (LTV) of customers to help businesses identify high-value customers.
- Use RFM features to engineer relevant customer metrics.
- Train a regression model (XGBoost) to predict CLV.
- Segment customers based on predicted CLV for marketing purposes.

## Tools Used
- **Python**:
    - Libraries: Pandas, Numpy, Matplotlib, Seaborn, XGBoost, Scikit-learn, Streamlit
    - Model: XGBoost Regressor
- **Excel**: For exporting filtered customer data
- **Streamlit**: For building the web application

## Key Features
1. **Model Evaluation**: 
    - Displays RMSE, MAE, and R² score for model evaluation.
    - Visualizes Actual vs Predicted CLV on a scatter plot.
    
2. **Top Customers by Predicted CLV**: 
    - Lists the top 10 customers based on their predicted CLV.
    
3. **Full RFM & CLV Table**:
    - Displays a complete table with customer-level RFM features and predicted CLV values.
    
4. **Export to Excel**:
    - Allows users to export the filtered customer data along with their CLV predictions into an Excel file.

## Steps to Run the App Locally

### 1. Install Dependencies:
Make sure you have Python 3.6+ installed. Then install the required libraries by running:

```bash
pip install -r requirements.txt
```

Where the `requirements.txt` file includes:
```
pandas
numpy
matplotlib
seaborn
xgboost
scikit-learn
streamlit
openpyxl
```

### 2. Prepare the Dataset:
Download the dataset from your source or use your own dataset (ensure it contains at least the following columns: `CustomerID`, `InvoiceNo`, `Quantity`, `UnitPrice`, `InvoiceDate`, and `Country`).

### 3. Run the Streamlit App:
To run the app locally, execute the following command in the terminal:
```bash
streamlit run app.py
```

This will launch the app in your browser.

### 4. Exporting Data:
You can filter and export customer data (with predicted CLV) by clicking the "Export to Excel" button in the app.

## Example Visualization

### Actual vs Predicted CLV
A scatter plot visualizes the performance of the model by comparing the actual vs. predicted CLV.

## Folder Structure
```
.
├── app.py                # Main Streamlit app
├── Online Retail.xlsx     # Customer transaction data (or your dataset)
├── filtered_rfm_clv.xlsx  # Exported file with filtered data and predicted CLV
└── requirements.txt       # Python dependencies file
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



