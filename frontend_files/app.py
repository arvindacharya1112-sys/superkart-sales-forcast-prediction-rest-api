
import pandas as pd
import requests
import streamlit as st

# Backend URL pointing to container name 'backend'
BACKEND_URL = "http://backend:7860"

st.title("SuperKart Sales Prediction")

# ==========================================
# 1. ONLINE PREDICTION SECTION
# ==========================================
st.subheader("Online Prediction")

# Input Controls
Product_Weight = st.number_input(
    "Product Weight (oz)", min_value=0.0, value=12.66
)
Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"],
    key="product_sugar_content",
)
Product_Allocated_Area = st.number_input(
    "Product Allocated Area", min_value=0.0, value=0.027
)
Product_MRP = st.number_input(
    "Maximum Retail Price (USD)", min_value=0.0, value=150.0
)
Store_Size = st.selectbox(
    "Store Size", ["Small", "Medium", "High"], key="store_size"
)
Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"],
    key="store_city_type",
)
Store_Type = st.selectbox(
    "Store Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Departmental Store",
        "Food Mart",
    ],
    key="store_type",
)

# Requested Features
Product_Id_char = st.text_input(
    "Product ID Char (e.g., FD, DR, NC)", value="FD", key="product_id_char"
)
Store_Age_Years = st.slider(
    "Store Age (years)", min_value=0, max_value=50, value=15
)

# Request Payload
payload = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
}

if st.button("Predict Online", type="primary", key="online_predict_btn"):
  try:
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
    if response.status_code == 200:
      predicted_sales = response.json().get("Predicted Sales (in dollars)")
      st.success(f"Predicted Sales: ${predicted_sales}")
    else:
      st.error(f"Backend Error ({response.status_code}): {response.text}")
  except requests.exceptions.RequestException as e:
    st.error(f"Unable to connect to prediction API: {e}")

st.markdown("---")

# ==========================================
# 2. BATCH PREDICTION SECTION
# ==========================================
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV for Batch Prediction", type=["csv"], key="batch_file_uploader"
)

if uploaded_file is not None:
  # Display preview of uploaded dataset
  try:
    df_preview = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_preview.head())

    # Reset file pointer so requests can read the file bytes
    uploaded_file.seek(0)
  except Exception as e:
    st.error(f"Error reading CSV file: {e}")

  # Explicit execution button for batch processing
  if st.button("Run Batch Prediction", key="batch_predict_btn", type="primary"):
    with st.spinner("Sending request to backend..."):
      try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")
        }

        response = requests.post(f"{BACKEND_URL}/v1/predict-batch", files=files)

        if response.status_code == 200:
          results = response.json()

          # Convert output dict to DataFrame
          df_results = pd.DataFrame(
              list(results.items()),
              columns=["ID / Index", "Predicted Sales ($)"],
          )

          st.success("Batch Prediction Complete!")
          st.dataframe(df_results)

          # Download button for batch results
          csv_data = df_results.to_csv(index=False).encode("utf-8")
          st.download_button(
              label="Download Predictions as CSV",
              data=csv_data,
              file_name="superkart_batch_predictions.csv",
              mime="text/csv",
          )
        else:
          st.error(f"API Error ({response.status_code}): {response.text}")

      except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to prediction API: {e}")
