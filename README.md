# Student Burnout Risk Classification (5 ML Models + Streamlit App)

## a. Problem Statement
Generative AI tools are now used by many students. The aim of this project is to predict a
student's **burnout risk level (Low / Medium / High)** from their academic background and AI usage
habits. This is a **multi-class classification** problem. Five classification models are built on
the same dataset, compared using six evaluation metrics, and demonstrated in a Streamlit web app.

## b. Dataset Description
- **Source:** Kaggle - "AI Impact on Students" dataset (file: `ai_student_impact_dataset.csv`)
- **Total instances in the file:** 50,000 students, 16 columns, no missing values, no duplicate rows
- **Instances used in this project:** a random sample of 10,000 rows (to keep SVM and KNN fast),
  split 80-20 into 8,000 training rows and 2,000 test rows (`test_data.csv`)
- **Number of input features:** 14 (the `Student_ID` column was removed because it is only a row number)
- **Target column:** `Burnout_Risk_Level` with 3 classes - Low, Medium, High
  (full file: Medium 21,144 / Low 16,369 / High 12,487 rows)

| Feature | Type |
|---|---|
| Major_Category | categorical (STEM, Business, Humanities, Medical, Arts) |
| Year_of_Study | categorical (Freshman ... Graduate) |
| Pre_Semester_GPA | numeric |
| Weekly_GenAI_Hours | numeric |
| Primary_Use_Case | categorical |
| Prompt_Engineering_Skill | categorical (Beginner, Intermediate, Advanced) |
| Tool_Diversity | numeric (1 to 5) |
| Paid_Subscription | True / False (converted to 1 / 0) |
| Traditional_Study_Hours | numeric |
| Perceived_AI_Dependency | numeric (1 to 10) |
| Institutional_Policy | categorical |
| Anxiety_Level_During_Exams | numeric (1 to 10) |
| Post_Semester_GPA | numeric |
| Skill_Retention_Score | numeric |

**Preprocessing:** numeric columns are scaled (z = (x - mean) / std), categorical columns are
one-hot encoded, and both steps are saved inside each model pipeline. The 80-20 split is stratified
so every part keeps the same class ratio.

## c. GitHub Repository Link
https://github.com/sunilbs05427-07/ai-student-burnout-classifier.git

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5380 | 0.6930 | 0.5528 | 0.5380 | 0.5387 | 0.2789 |
| Decision Tree | 0.4400 | 0.5673 | 0.4402 | 0.4400 | 0.4400 | 0.1425 |
| KNN | 0.4370 | 0.6026 | 0.4376 | 0.4370 | 0.4362 | 0.1433 |
| Naive Bayes | 0.5065 | 0.6715 | 0.5127 | 0.5065 | 0.4998 | 0.2506 |
| Support Vector Machine | 0.5120 | 0.6727 | 0.5377 | 0.5120 | 0.5090 | 0.2324 |

Precision, Recall and F1 are weighted averages over the 3 classes (Low, Medium, High). AUC is one-vs-rest, weighted. All values are calculated on the 2,000-row test set (`test_data.csv`).

Precision, Recall and F1 are weighted averages over the 3 classes. AUC is one-vs-rest, weighted.

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gave the best result on all six metrics (Accuracy 0.538, AUC 0.693, F1 0.5387, MCC 0.2789). A simple linear model performed better than the more complex ones, so the relationship between the features and burnout risk seems mostly linear and noisy. It is also fast to train and easy to interpret. |
| Decision Tree | Weakest model on Accuracy, Precision and F1 (Accuracy 0.44, MCC 0.1425), only about 2 points above the 42% baseline. AUC is also the lowest (0.5673). An unrestricted tree tends to memorize the training data and does not generalize well to new students. |
| KNN | Performed similarly to the Decision Tree (Accuracy 0.437, MCC 0.1433), but its AUC (0.6026) is better. Distance-based methods struggle when many features are noisy and the one-hot encoding adds extra columns, so the neighbours are not very informative. |
| Naive Bayes | Second best on MCC (0.2506) and close to the SVM (Accuracy 0.5065, AUC 0.6715). It works reasonably well even with the assumption that features are independent, and it is the fastest model to train. |
| Support Vector Machine | Second best on Accuracy (0.512) and Precision (0.5377), with AUC 0.6727, but its MCC (0.2324) is lower than Naive Bayes. The RBF kernel did not improve on Logistic Regression, so there is little non-linear pattern to learn. It was also the slowest model to train. |

**Overall winner for this dataset: Logistic Regression.**
It has the highest score on every metric (Accuracy 0.538, AUC 0.693, Precision 0.5528, Recall 0.538, F1 0.5387, MCC 0.2789). It is also the simplest model and trains quickly. Naive Bayes and SVM are close behind, while Decision Tree and KNN are clearly the weakest. Overall scores are modest (MCC about 0.28), so the available features only partly predict burnout risk.

## Streamlit App
Live app link: https://ai-student-burnout-classifier-sunilbs05427.streamlit.app/

How to use: upload `test_data.csv`, pick a model from the dropdown, and view the metrics,
confusion matrix and classification report. The saved models are loaded, not retrained.

## How to run locally
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
