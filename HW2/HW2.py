# Understanding the Problem Statement:
# The goal is to predict whether a given molecule is active (1) or inactive (0) by using data-driven techniques, specifically binary classification models. In drug discovery, identifying whether a molecule binds to a specific receptor is a crucial step since binding generally indicates the molecule may be a good candidate for a drug.

# Overview and Assignment Goals
# The objectives of this assignment are the following:

# Use/implement a feature selection/reduction technique.
# Experiment with various classification models.
# Think about dealing with imbalanced data.
# Use F1 Scoring Metric
# The dataset has an imbalanced distribution i.e., within the training set there are only 78 actives (+1) and 722 inactives (0). No information is provided for the test set regarding the distribution.
# Data Description
# The training dataset consists of 800 records and the test dataset consists of 350 records. We provide you with the training class labels and the test labels are held out. The attributes are binary type and as such are presented in a sparse matrix format within train.dat and test.dat

# Train data: Training set (a sparse binary matrix, patterns in lines, features in columns: the index of the non-zero features are provided with class label 1 or 0 in the first column).

# Test data: Testing set (a sparse binary matrix, patterns in lines, features in columns: the index of non-zero features are provided).

# Format example: A sample submission with 350 entries randomly chosen to be 0 or 1.


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline

results = []
# Function to load and process data from TXT files
def load_and_process_data(file_path, is_train=True):
    data = []
    labels = [] if is_train else None

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parts = line.strip().split()
                if is_train:
                    labels.append(int(parts[0]))  # Extract class label
                    features = list(map(int, parts[1:]))  # Extract feature indices
                else:
                    features = list(map(int, parts))  # Only feature indices for test data
                data.append(features)

    max_index = max(max(features) for features in data)
    binary_array = np.zeros((len(data), max_index + 1), dtype=int)

    for row_idx, indices in enumerate(data):
        binary_array[row_idx, indices] = 1

    csr_feature_matrix = csr_matrix(binary_array)
    if is_train:
        return csr_feature_matrix, np.array(labels)
    else:
        return csr_feature_matrix

# Preprocessing pipeline
pre_processing_pipeline = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.03)),
    ('svd', TruncatedSVD(n_components=80, random_state=42)),
    ('scaler', MaxAbsScaler())
])

# List of imbalance handling techniques
resampling_techniques = {
    'RandomOversampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'RandomUndersampling': RandomUnderSampler(random_state=42)
}

# Hyperparameter grid for Decision Tree
param_grid = {
    'classifier__max_depth': range(2, 10),
    'classifier__min_samples_leaf': range(1, 6),
    'classifier__min_samples_split': range(2, 6)
}

# Load and preprocess training data
X, y = load_and_process_data('train_data.txt', is_train=True)
X_preprocessed = pre_processing_pipeline.fit_transform(X)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Function to train and predict with different imbalance techniques
def train_and_predict_with_resampling(technique_name, resampler):
    pipeline = Pipeline([
        ('resampler', resampler),
        ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ])
    
    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=4, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    f1 = f1_score(y_val, y_val_pred)
    results.append([technique_name, f1, grid_search.best_params_])

    # Display results
    print(f"\nUsing {technique_name}:")
    print("Best parameters:", grid_search.best_params_)
    print(f"F1-Score (Validation): {f1_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred))

    # Load and preprocess the test data
    X_test = load_and_process_data('test_data.txt', is_train=False)
    X_test_preprocessed = pre_processing_pipeline.transform(X_test)

    # Predict on the test data
    test_predictions = best_model.predict(X_test_preprocessed)

    # Save predictions to a TXT file
    output_file = f"format_file_{technique_name}.txt"
    with open(output_file, 'w+') as file:
        for prediction in test_predictions:
            file.write(str(prediction) + '\n')

    print(f"Predictions saved to {output_file}")

# Apply all resampling techniques and predict
for technique_name, resampler in resampling_techniques.items():
    train_and_predict_with_resampling(technique_name, resampler)


df_results = pd.DataFrame(results, columns=['Technique', 'F1-Score', 'Best Parameters'])

# Visualization of F1-scores using a bar chart
plt.figure(figsize=(8, 4))
plt.barh(df_results['Technique'], df_results['F1-Score'], color='skyblue')
plt.xlabel('F1-Score')
plt.title('F1-Score Comparison for Different Resampling Techniques')
plt.gca().invert_yaxis()  # Invert y-axis to show highest score on top
plt.show()
