from sklearn.metrics import confusion_matrix, recall_score, average_precision_score, precision_recall_curve
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

df = pd.read_csv(r"C:\Users\User\Desktop\creditcard (1).csv")







y = df["Class"]
x = df.drop(columns=["Class"])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
#
# ml = RandomForestClassifier(random_state=42,class_weight="balanced_subsample",n_jobs=1,bootstrap=True)
# ml.fit(x_train, y_train)
# y_pred = ml.predict(x_test)
# y_probs = ml.predict_proba(x_test)[:, 1]


# print(confusion_matrix(y_test, y_pred))

# print(f"Recall): {recall_score(y_test, y_pred):.2%}")


# print(f"PR-AUC: {average_precision_score(y_test, y_probs):.4f}")





x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train_val,y_train_val,test_size=0.1,random_state=42,stratify=y_train_val)
ml = RandomForestClassifier(random_state=42,class_weight="balanced_subsample",n_jobs=1,bootstrap=True)

param_grid = {
'n_estimators':[200],
'max_depth':[8,10,12],
'min_samples_leaf':[3],
'max_features': ['sqrt'],
"bootstrap": [True],}




grid_search = GridSearchCV(ml,param_grid=param_grid,scoring='average_precision',cv=3,verbose=2,n_jobs=-1)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
y_probs = best_model.predict_proba(x_test)[:, 1]

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Recall на TEST: {recall_score(y_test, y_pred):.4f}")
print(f"PR-AUC на TEST: {average_precision_score(y_test, y_probs):.4f}")
print(confusion_matrix(y_test, y_pred))




y_probs_val = best_model.predict_proba(x_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs_val)
target_recall = 0.80
valid_indices = np.where(recalls[:-1] >= target_recall)[0]
best_idx = valid_indices[np.argmax(precisions[:-1][valid_indices])]
best_threshold = thresholds[best_idx]

y_probs_test = best_model.predict_proba(x_test)[:, 1]
y_pred_custom = (y_probs_test >= best_threshold).astype(int)

print(f"Recall на TEST: {recall_score(y_test, y_pred_custom):.4f}")
print(f"PR-AUC на TEST: {average_precision_score(y_test, y_probs_test):.4f}")
print(confusion_matrix(y_test, y_pred_custom))