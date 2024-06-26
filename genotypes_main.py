import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import os

# Define column names for the datasets
column_names = ['id ref value', 'log2 SST-RMA', 'title', 'Sample type', 'source name', 'organism', 'tissue', 'genotype', 'age', 'sex', 'diagnosis']

# Specify dtypes to avoid mixed type warning
dtype_spec = {
    'id ref value': str,
    'log2 SST-RMA': float,
    'title': str,
    'Sample type': str,
    'source name': str,
    'organism': str,
    'tissue': str,
    'genotype': str,
    'age': float,  # Assuming age is numeric
    'sex': str,
    'diagnosis': str
}

# Load datasets with appropriate dtype handling
print("Loading datasets...")
blood_data = pd.read_csv('/content/drive/MyDrive/bloodsamplefinal.csv', names=column_names, dtype=dtype_spec, skiprows=1)
print("Blood data loaded.")
bone_marrow_data = pd.read_csv('/content/drive/MyDrive/bonemarrowfinal.csv', names=column_names, dtype=dtype_spec, skiprows=1)
print("Bone marrow data loaded.")

# Concatenate both datasets
print("Concatenating datasets...")
data = pd.concat([blood_data, bone_marrow_data], ignore_index=True)
print(f"Total records: {data.shape[0]}")

# Fill missing values using forward fill method
print("Filling missing values...")
data = data.fillna(method='ffill')
print("Missing values filled.")

# Separate features and target variable
print("Separating features and target...")
features = data.drop(['id ref value', 'diagnosis'], axis=1)
target = data['diagnosis']

# One-hot encoding for categorical columns
print("One-hot encoding categorical columns...")
categorical_columns = ['title', 'Sample type', 'source name', 'organism', 'tissue', 'genotype', 'sex']
features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)
print("Encoding complete.")

# Standard scaling on the features
print("Standard scaling features...")
scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)
normalized_data = pd.DataFrame(normalized_data, columns=features.columns)

# Add the target column back to the normalized data
normalized_data['diagnosis'] = target
print("Scaling complete.")

# Split the data into training and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(normalized_data.drop('diagnosis', axis=1), normalized_data['diagnosis'], test_size=0.2, random_state=42)
print("Data split complete.")

# Random Forest model with parallel processing
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained.")

# Support Vector Machine model with probability estimates
print("Training SVM model...")
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
print("SVM model trained.")

# Model Evaluation
print("Evaluating models...")
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

print('Random Forest Classification Report:')
print(classification_report(y_test, rf_predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, rf_predictions))

print('\nSVM Classification Report:')
print(classification_report(y_test, svm_predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, svm_predictions))

# Compute ROC AUC Scores
print("Computing ROC AUC scores...")
rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
svm_roc_auc = roc_auc_score(y_test, svm_model.decision_function(X_test))
print(f'\nRandom Forest ROC AUC Score: {rf_roc_auc}')
print(f'SVM ROC AUC Score: {svm_roc_auc}')

# Model Refinement using Randomized Search for Random Forest
print("Running Randomized Search for Random Forest...")
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(rf_model, param_distributions, n_iter=20, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
print('\nBest Parameters:', random_search.best_params_)

# Use best model
best_rf_model = random_search.best_estimator_

# Final Model Evaluation with Best Parameters
print("Evaluating best Random Forest model...")
best_rf_predictions = best_rf_model.predict(X_test)
print('\nBest Random Forest Classification Report:')
print(classification_report(y_test, best_rf_predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, best_rf_predictions))

best_rf_roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1])
print(f'Best Random Forest ROC AUC Score: {best_rf_roc_auc}')

# Directory for saving plots
print("Generating plots...")
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Plot ROC curves and save
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, name='Random Forest')
RocCurveDisplay.from_estimator(svm_model, X_test, y_test, ax=ax, name='SVM')
plt.title('ROC Curves')
plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
plt.close(fig)

# Plot and save confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_rf = confusion_matrix(y_test, rf_predictions)
cm_svm = confusion_matrix(y_test, svm_predictions)

for ax, cm, title in zip(axes, [cm_rf, cm_svm], ['Random Forest', 'SVM']):
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
plt.close(fig)
print("Plots generated and saved.")
