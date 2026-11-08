"""
app.py  -  Streamlit app for Assignment 2
Loads the SAVED models (no retraining) and evaluates them on an uploaded test csv.
"""

# STEP 1: Import libraries
import joblib                                        # to load the saved models
import pandas as pd                                  # to read the uploaded csv
import streamlit as st                               # the web app library
import matplotlib.pyplot as plt                      # to draw the confusion matrix
import seaborn as sns                                # heatmap for the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
from metrics_helper import calculate_metrics         # same metric function as in training

# STEP 2: Page settings
st.set_page_config(page_title="Classification Model Demo", layout="wide")
st.title("Classification Models - Test Data Evaluation")
st.write("Upload the test csv, choose a model and see its performance.")

# STEP 3: Dictionary of model names and their saved files
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource                                   # load each file only once (faster app)
def load_file(path):
    return joblib.load(path)


label_encoder = load_file("model/label_encoder.joblib")   # class names <-> numbers
target_col = load_file("model/target_column.joblib")      # name of the target column

# STEP 4: Sidebar - upload option and model dropdown
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

# STEP 5: Stop here until a file is uploaded
if uploaded_file is None:
    st.info("Please upload the test csv file from the sidebar.")
    st.stop()

# STEP 6: Read the csv and check that the target column is present
data = pd.read_csv(uploaded_file)
st.subheader("Uploaded data preview")
st.dataframe(data.head())

if target_col not in data.columns:
    st.error("The target column '" + target_col + "' is missing in the uploaded file.")
    st.stop()

# Separate features and target; convert target labels to numbers
X_test = data.drop(columns=[target_col])
try:
    y_test = label_encoder.transform(data[target_col])
except ValueError:
    st.error("The target column has class values that the model has not seen.")
    st.stop()

# STEP 7: Metrics of the selected model
model = load_file(MODEL_FILES[model_name])           # load the saved (already trained) model
scores = calculate_metrics(model, X_test, y_test)

st.subheader("Evaluation metrics - " + model_name)
cols = st.columns(6)                                 # six boxes in one row
for col, (metric, value) in zip(cols, scores.items()):
    col.metric(metric, round(value, 4))

# STEP 8: Confusion matrix and classification report
y_pred = model.predict(X_test)
class_names = [str(c) for c in label_encoder.classes_]

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    st.text(report)

# STEP 9: Comparison of all 5 models on the uploaded data
st.subheader("Comparison of all models on this test data")
rows = []
for name, path in MODEL_FILES.items():
    row = calculate_metrics(load_file(path), X_test, y_test)
    row["ML Model Name"] = name
    rows.append(row)
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comparison)