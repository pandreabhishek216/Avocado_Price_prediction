import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.immediate.co.uk/production/volatile/sites/30/2022/07/Avocado-sliced-in-half-ca9d808.jpg?quality=90&webp=true&resize=440,400');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .overlay {
        background: rgba(0,0,0,0.65);
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }
    h1, h2, h3, h4, p, label {
        color: white!important;
    }
    </style>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv("avocado.csv")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["year"] = df["Date"].dt.year
df = df.drop("Date", axis=1)

le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

drop_cols = [col for col in ["region", "Unnamed: 0"] if col in df.columns]
df = df.drop(drop_cols, axis=1)

df_major = df[df.type == 0]
df_minor = df[df.type == 1]

df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
df_balanced = pd.concat([df_major, df_minor_up])

X = df_balanced.drop("type", axis=1)
y = df_balanced["type"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report_dict = classification_report(
    y_test, y_pred, target_names=le.classes_, output_dict=True
)

st.title("Avocado Type Prediction")
st.subheader("Model Performance")

performance_table = pd.DataFrame(report_dict).transpose()
st.dataframe(performance_table.style.highlight_max(axis=0), height=350)

st.subheader("Make a Prediction")

inputs = []
for col in feature_names:
    val = st.number_input(col, value=float(df[col].mean()))
    inputs.append(val)

if st.button("Predict"):
    arr = np.array([inputs])
    prediction = model.predict(arr)
    predicted_type = le.inverse_transform(prediction)[0]
    proba = model.predict_proba(arr)[0]

    st.subheader(f"Predicted type: {predicted_type}")
    st.write("Prediction probabilities:", proba)

    fig, ax = plt.subplots(figsize=(6, 4))

    if predicted_type.lower() == "conventional":
        graph_values = {
            "AveragePrice": arr[0][feature_names.index("AveragePrice")],
            "Total Volume": arr[0][feature_names.index("Total Volume")],
            "Total Bags": arr[0][feature_names.index("Total Bags")]
        }
        ax.bar(graph_values.keys(), graph_values.values())
        ax.set_title("Conventional Avocado Stats")

    else:
        organic_avg = {
            "AveragePrice": 1.48,
            "Total Volume": 7920500,
            "Total Bags": 4801233
        }
        ax.plot(list(organic_avg.keys()), list(organic_avg.values()), marker="o")
        ax.set_title("Organic Avocado Trend")

    ax.set_ylabel("Value")
    st.pyplot(fig)
