"""
metrics_helper.py
Common function used by BOTH train_models.py and app.py,
so the 6 evaluation metrics are always calculated in the same way.
"""

# numpy is used for simple array operations
import numpy as np
# these are the 6 metrics asked in the assignment
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def calculate_metrics(model, X, y_true):
    """
    model  : a trained pipeline (preprocessing + classifier)
    X      : input features (data frame)
    y_true : actual class labels (already encoded as 0, 1, 2, ...)
    returns: dictionary with Accuracy, AUC, Precision, Recall, F1, MCC
    """
    # predicted class for every row
    y_pred = model.predict(X)
    # predicted probability of every class (needed for the AUC score)
    y_prob = model.predict_proba(X)

    # number of classes = number of probability columns
    n_classes = y_prob.shape[1]

    if n_classes == 2:
        # Binary problem: use the probability of the positive class (column 1)
        avg = "binary"
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        # Multi-class problem: one-vs-rest AUC and weighted average of the classes
        avg = "weighted"
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")

    # Formulae (binary case):
    # Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    # Precision = TP / (TP + FP)
    # Recall    = TP / (TP + FN)
    # F1        = 2 x Precision x Recall / (Precision + Recall)
    # MCC       = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return results
