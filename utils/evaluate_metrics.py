import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the true labels
true_df = pd.read_csv("../labeled_service/mobservice2.csv")
true_df = true_df[["timestamp", "label"]]

# Load the predictions
pred_df = pd.read_csv("mobservice2_ed.csv")
pred_df = pred_df.rename(columns={"timetamp": "timestamp"})  # fix typo
pred_df["label"] = pred_df["label"].apply(lambda x: int(x.strip("[]")))  # convert '[1]' -> 1

# Merge on timestamp to align predictions and ground truth
merged_df = pd.merge(pred_df, true_df, on="timestamp", suffixes=("_pred", "_true"))

# Extract predictions and ground truth labels
y_pred = merged_df["label_pred"]
y_true = merged_df["label_true"]

# Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Output results
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
