import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("📊 Telco Customer Churn Prediction")
st.write("Provide the customer details below to predict the likelihood of churn.")

# ----------------------------- FORM START -----------------------------------
with st.form("customer_form"):

    st.subheader("👤 Customer Information")
    gender = st.radio("Gender", ["Male", "Female"])
    senior_citizen = st.radio("Senior Citizen", ["0", "1"])
    partner = st.radio("Partner", ["Yes", "No"])
    dependents = st.radio("Dependents", ["Yes", "No"])

    st.subheader("📞 Customer Service Details")
    phone_service = st.radio("Phone Service", ["Yes", "No"])
    multiple_lines = st.radio("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.radio("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.radio("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.radio("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.radio("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.radio("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.radio("Streaming Movies", ["Yes", "No", "No internet service"])

    st.subheader("💳 Billing Information")
    contract = st.radio("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
    payment_method = st.radio("Payment Method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])

    st.subheader("💰 Charges")
    tenure = st.number_input("Tenure Months", min_value=0, max_value=72, step=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    submitted = st.form_submit_button("🔍 Predict Churn")

# ----------------------------- FORM END -----------------------------------

if submitted:
    # Convert categorical to numerical as per encoded model
    def encode_yes_no(value):
        return 1 if value == "Yes" else 0

    # Customer info encoding
    gender = 1 if gender == "Male" else 0
    senior_citizen = int(senior_citizen)
    partner = encode_yes_no(partner)
    dependents = encode_yes_no(dependents)

    phone_service = encode_yes_no(phone_service)
    multiple_lines = 2 if multiple_lines == "No phone service" else encode_yes_no(multiple_lines)

    # Internet service mapping
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    internet_service = internet_map[internet_service]

    # Other services
    def encode_service(x):
        return 2 if x == "No internet service" else encode_yes_no(x)

    online_security = encode_service(online_security)
    online_backup = encode_service(online_backup)
    device_protection = encode_service(device_protection)
    tech_support = encode_service(tech_support)
    streaming_tv = encode_service(streaming_tv)
    streaming_movies = encode_service(streaming_movies)

    # Contract mapping
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract = contract_map[contract]

    # Payment method mapping
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
    payment_method = payment_map[payment_method]

    paperless_billing = encode_yes_no(paperless_billing)

    # Prepare input vector
    input_data = np.array([[gender, senior_citizen, partner, dependents,
                            tenure, phone_service, multiple_lines, internet_service,
                            online_security, online_backup, device_protection,
                            tech_support, streaming_tv, streaming_movies,
                            contract, paperless_billing, payment_method,
                            monthly_charges, total_charges]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    st.subheader("📢 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
