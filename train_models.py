"""
train_models.py
Assignment 2 : trains 5 classification models on ONE dataset, calculates
6 evaluation metrics for each, and saves the models for the Streamlit app.

Split rule (80-20):
    N          = total rows in the dataset
    test_size  = 0.20 x N   -> saved as test_data.csv
    train_size = N - test_size (80%) -> used to train the models
"""

# ---------------------------------------------------------------
# STEP 1: Import libraries
# ---------------------------------------------------------------
import os                                            # to create the model folder
import joblib                                        # to save trained models as files
import pandas as pd                                  # to handle the data table
from sklearn.model_selection import train_test_split # for the 80-20 split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer             # to fill missing values
from sklearn.compose import ColumnTransformer        # to treat number / text columns differently
from sklearn.pipeline import Pipeline               # to join preprocessing + model in one object
from sklearn.linear_model import LogisticRegression  # model 1
from sklearn.tree import DecisionTreeClassifier      # model 2
from sklearn.neighbors import KNeighborsClassifier   # model 3
from sklearn.naive_bayes import GaussianNB           # model 4
from sklearn.svm import SVC                          # model 5
from metrics_helper import calculate_metrics         # our own metric function

# ---------------------------------------------------------------
# STEP 2: Initial Settings  
# ---------------------------------------------------------------
DATA_PATH = "ai_student_impact_dataset.csv"   # path of the downloaded Kaggle csv file
TARGET_COL = "Burnout_Risk_Level"            # name of the column we want to predict (Low / Medium / High)
DROP_COLS = ["Student_ID"]                   # ID column is only a row number, so we remove it
SAMPLE_ROWS = 10000          # use only this many random rows (SVM and KNN are slow on 50,000); None = use all
TEST_RATIO = 0.20            # 20% data kept aside for testing
RANDOM_SEED = 36
            # same shuffle every time we run
MODEL_FOLDER = "model"       # folder where models will be saved

# ---------------------------------------------------------------
# STEP 3: Loading the dataset and check conditions
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)                          # read csv into a data frame
df = df.drop(columns=DROP_COLS)                      # remove the unwanted columns
df = df.dropna(subset=[TARGET_COL])                  # rows without a target are useless

# Use a random sample of the rows so that training finishes quickly
if SAMPLE_ROWS is not None and len(df) > SAMPLE_ROWS:
    df = df.sample(n=SAMPLE_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

# True/False columns are converted to 1/0 so they are treated as numbers
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

n_rows = df.shape[0]                                 # number of instances
n_features = df.shape[1] - 1                         # all columns except the target
print("Instances:", n_rows, "| Features:", n_features)

# The assignment needs at least 500 instances and 12 features
if n_rows < 500 or n_features < 12:
    print("WARNING: dataset is smaller than the assignment minimum (500 rows, 12 features)")

# ---------------------------------------------------------------
# STEP 4: Convert the target labels into numbers (0, 1, 2, ...)
# ---------------------------------------------------------------
label_encoder = LabelEncoder()                       # object that maps class names to numbers
y_all = label_encoder.fit_transform(df[TARGET_COL])  # learn the classes and convert
print("Classes:", list(label_encoder.classes_))

# ---------------------------------------------------------------
# STEP 5: 80-20 split (stratify keeps the class ratio same in both parts)
# ---------------------------------------------------------------
train_df, test_df = train_test_split(
    df, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y_all
)
print("Train rows:", len(train_df), "| Test rows:", len(test_df))

# Save the raw test part (features + target) for the GitHub repo and Streamlit app
test_df.to_csv("test_data.csv", index=False)

# Separate features (X) and target (y) for training
X_train = train_df.drop(columns=[TARGET_COL])
y_train = label_encoder.transform(train_df[TARGET_COL])
X_test = test_df.drop(columns=[TARGET_COL])
y_test = label_encoder.transform(test_df[TARGET_COL])


# ---------------------------------------------------------------
# STEP 6: Preprocessing function
# ---------------------------------------------------------------
def make_preprocessor(X):
    """Numbers: fill missing with median + scale. Text: fill missing + one-hot encode."""
    num_cols = X.select_dtypes(include="number").columns      # numeric columns
    cat_cols = X.select_dtypes(exclude="number").columns      # text / category columns

    num_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median")),           # missing number -> median
        ("scale", StandardScaler()),                          # z = (x - mean) / std
    ])
    cat_steps = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),    # missing text -> most common value
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_steps, num_cols),
        ("cat", cat_steps, cat_cols),
    ])


# ---------------------------------------------------------------
# STEP 7: The 5 models required
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_SEED),  # probability=True is needed for AUC
}

# ---------------------------------------------------------------
# STEP 8: Train, evaluate and save every model
# ---------------------------------------------------------------
os.makedirs(MODEL_FOLDER, exist_ok=True)             # create model/ folder if not present
all_results = []                                     # list to collect the metrics of each model

for name, clf in models.items():
    # join preprocessing and classifier so the app can use raw csv data directly
    pipe = Pipeline([("prep", make_preprocessor(X_train)), ("clf", clf)])

    pipe.fit(X_train, y_train)                       # train ONLY on the 80% training data

    scores = calculate_metrics(pipe, X_test, y_test) # evaluate on the 20% test data
    scores["ML Model Name"] = name                   # add the model name to the row
    all_results.append(scores)

    # file name e.g. "Logistic Regression" -> model/logistic_regression.joblib
    file_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_FOLDER, file_name))
    print("Saved:", file_name)

# Save the other things the app needs
joblib.dump(label_encoder, os.path.join(MODEL_FOLDER, "label_encoder.joblib"))
joblib.dump(TARGET_COL, os.path.join(MODEL_FOLDER, "target_column.joblib"))

# ---------------------------------------------------------------
# STEP 9: Comparison table (copy this into your README)
# ---------------------------------------------------------------
table = pd.DataFrame(all_results)
table = table[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
table = table.round(4)                               # 4 decimal places
table.to_csv(os.path.join(MODEL_FOLDER, "comparison_table.csv"), index=False)
print(table.to_string(index=False))
