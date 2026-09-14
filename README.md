Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using Random Forest.

What I did

• Worked with highly imbalanced data
• Train/Validation/Test split
• Random Forest classification
• Hyperparameter tuning with GridSearchCV
• Evaluation using Recall and PR-AUC
• Confusion Matrix analysis
• Threshold optimization using the validation set
• Target Recall optimization to reduce false negatives

Technologies

Python · Pandas · NumPy · Scikit-learn

Model

Random Forest Classifier

Threshold Optimization

The classification threshold was selected on the validation set based on a target Recall of 80% and then evaluated on the unseen test set.

Dataset

The dataset was obtained from the following GitHub repository:

Credit Card Fraud Detection Dataset
https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection

The dataset contains credit card transactions labeled as fraudulent or legitimate.
