import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


def load_pickles(model_pickle_path, label_encoder_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)

    with open(label_encoder_pickle_path, "rb") as label_encoder_opener:
        label_encoder_dict = pickle.load(label_encoder_opener)

    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df_out = df.copy()
    df_out.replace(" ", 0, inplace=True)
    df_out.loc[:, "TotalCharges"] = pd.to_numeric(df_out.loc[:, "TotalCharges"])
    if "customerID" in df.columns:
        df_out.drop("customerID", axis=1, inplace=True)
    for column, le in label_encoder_dict.items():
        if column in df_out.columns:
            df_out.loc[:, column] = le.transform(df.loc[:, column])

    return df_out


def make_predictions(test_data):
    model_pickle_path = "./models/churn_prediction_model.pkl"
    label_encoder_pickle_path = "./models/churn_prediction_label_encoder.pkl"

    model, label_encoder_dict = load_pickles(
        model_pickle_path, label_encoder_pickle_path
    )

    data_processed = pre_process_data(test_data, label_encoder_dict)
    if "Churn" in test_data.columns:
        data_processed = data_processed.drop(columns=["Churn"])

    prediction = model.predict(data_processed)
    return prediction


if __name__ == "__main__":
    st.title("Customer churn predict")
    data = pd.read_csv("./data/holdout_data.csv")

    # visualise customer's data
    # st.text("Select Customer")
    customer = st.selectbox(
        label="Select Customer",
        options=data["customerID"],
    )

    customer_data = data[data["customerID"] == customer]
    st.table(customer_data)

    # Ask for customer's data

    gender = st.selectbox(label="Gender", options=["Male", "Female"])

    senior_citizen_input = st.selectbox(
        label="Is the customer a senior citizien?", options=["yes", "no"]
    )
    senior_citizen = 1 if senior_citizen_input == 1 else 0

    partner = st.selectbox(
        label="Is the customer a partner",
        options=[values for values in set(data.loc[:, "Partner"].values)],
    )
    dependents = st.selectbox(
        label="Is the customer a dependent?",
        options=[values for values in set(data.loc[:, "Dependents"].values)],
    )

    tenure = st.slider(
        label="How many months has the customer been with the company?",
        min_value=0,
        max_value=72,
        value=24,
    )

    phone_service = st.selectbox(
        label="Phone Service?",
        options=[values for values in set(data.loc[:, "PhoneService"].values)],
    )
    multiple_lines = st.selectbox(
        label="Multiple lines?",
        options=[values for values in set(data.loc[:, "MultipleLines"].values)],
    )
    internet_service = st.selectbox(
        label="Internet service?",
        options=[values for values in set(data.loc[:, "InternetService"].values)],
    )
    online_security = st.selectbox(
        label="Online security?",
        options=[values for values in set(data.loc[:, "OnlineSecurity"].values)],
    )
    online_backup = st.selectbox(
        label="Online backup?",
        options=[values for values in set(data.loc[:, "OnlineBackup"].values)],
    )
    device_protection = st.selectbox(
        label="Device protection?",
        options=[values for values in set(data.loc[:, "DeviceProtection"].values)],
    )
    TechSupport = st.selectbox(
        label="Tech support?",
        options=[values for values in set(data.loc[:, "TechSupport"].values)],
    )
    streaming_tv = st.selectbox(
        label="Streaming tv?",
        options=[values for values in set(data.loc[:, "StreamingTV"].values)],
    )
    streaming_movies = st.selectbox(
        label="Streaming movies?",
        options=[values for values in set(data.loc[:, "StreamingMovies"].values)],
    )
    contract = st.selectbox(
        label="Contract?",
        options=[values for values in set(data.loc[:, "Contract"].values)],
    )
    paperless_billing = st.selectbox(
        label="Paperless billing?",
        options=[values for values in set(data.loc[:, "PaperlessBilling"].values)],
    )
    payment_method = st.selectbox(
        label="Payement method?",
        options=[values for values in set(data.loc[:, "PaymentMethod"].values)],
    )
    monthly_charges = st.slider(
        label="Monthly charges:", min_value=0, max_value=200, value=5
    )
    total_charges = st.slider(
        label="Total charges:", min_value=0, max_value=8000, value=1000
    )

    customer_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": TechSupport,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    customer_data = pd.DataFrame([customer_dict])
    st.table(customer_data)

    if st.button("Predict Churn"):
        prediction = make_predictions(data)[0]
        prediction_string = "will churn" if prediction == 1 else "won't churn"
        st.text(f"Customer prediction: {prediction_string}")


#       or pd.DataFrame([customer_dict])

# data : pd.read_csv("./data/training_data.csv", index_col:0)
# st.write("Telco Churn Data")
# # st.write(data)
# # st.table(data)
# st.dataframe(data)


# # st.write("Here we create data using a table:")
# # st.write(pd.DataFrame({
# #     'first column': [1, 2, 3, 4],
# #     'second column': [10, 20, 30, 40]
# # }))


# st.write("How many customers in the dataset churned?")
# target_counts : data["Churn"].value_counts()
# # st.bar_chart(target_counts)
# # st.plotly_chart(data)

# st.bar_chart(data.groupby(["gender", "Contract"]).count().iloc[:, [0, 1]])
