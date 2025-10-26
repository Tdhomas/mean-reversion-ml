#model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import json
import os

warnings.filterwarnings('ignore')

from pathlib import Path

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PARAMS_DIR = Path("params")
PLOTS_DIR = Path("plots")

# Create output dirs on demand when the script writes files
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Define the combined strategy label
LABEL_NAME = 'Combined_Strategy'

def load_best_parameters(filepath= PARAMS_DIR / 'best_parameters.json'):
    """
    Load the best parameters for the combined strategy from a JSON file.
    If the file or strategy does not exist, return None.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as file:
        params = json.load(file)
    return params.get(LABEL_NAME, None)

def save_best_parameters(best_params_dict, filepath= PARAMS_DIR / 'best_parameters.json'):
    """
    Save the best parameters for the combined strategy to a JSON file.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            existing_params = json.load(file)
    else:
        existing_params = {}
    existing_params.update(best_params_dict)
    with open(filepath, 'w') as file:
        json.dump(existing_params, file, indent=4)

def iterative_hyperparameter_search(features, labels, initial_param_dist, max_iterations=3, n_iter=10, cv=5, scoring='precision', random_state=42):
    """
    Perform iterative hyperparameter search using RandomizedSearchCV.
    
    Parameters:
    - features (pd.DataFrame): Feature set.
    - labels (pd.Series): Labels.
    - initial_param_dist (dict): Initial parameter distribution for RandomizedSearchCV.
    - max_iterations (int): Maximum number of iterations for refining the search.
    - n_iter (int): Number of parameter settings sampled in each RandomizedSearchCV.
    - cv (int): Number of cross-validation splits.
    - scoring (str): Scoring metric for evaluation.
    - random_state (int): Random seed for reproducibility.
    
    Returns:
    - best_params (dict): Best hyperparameters found.
    - best_score (float): Best score achieved.
    """
    best_params = {}
    best_score = -np.inf
    param_dist = initial_param_dist.copy()

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1} of hyperparameter search.")
        rf_classifier = RandomForestClassifier(random_state=random_state, class_weight='balanced')

        random_search = RandomizedSearchCV(
            estimator=rf_classifier,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=cv),
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )

        random_search.fit(features, labels)
        iteration_best_score = random_search.best_score_
        iteration_best_params = random_search.best_params_

        print(f"Iteration {iteration + 1} - Best {scoring}: {iteration_best_score:.4f}")
        print(f"Iteration {iteration + 1} - Best Parameters: {iteration_best_params}")

        # Check for improvement
        if iteration_best_score > best_score:
            best_score = iteration_best_score
            best_params = iteration_best_params

            # Refine the parameter distribution for the next iteration
            param_dist = refine_param_dist(param_dist, iteration_best_params)
        else:
            print("No improvement in this iteration. Stopping hyperparameter search.")
            break

    return best_params, best_score

def refine_param_dist(current_param_dist, best_params):
    """
    Refine the parameter distribution around the current best parameters.
    This function narrows the search space based on the best parameters found.
    
    Parameters:
    - current_param_dist (dict): Current parameter distribution.
    - best_params (dict): Best parameters found in the current iteration.
    
    Returns:
    - new_param_dist (dict): Refined parameter distribution for the next iteration.
    """
    new_param_dist = {}
    for param, value in best_params.items():
        if param == 'n_estimators':
            new_param_dist[param] = [max(50, value - 50), value, value + 50]
        elif param == 'max_depth':
            if value is None:
                new_param_dist[param] = [None, 10, 20]
            else:
                new_param_dist[param] = [max(1, value - 5), value, value + 5]
        elif param == 'min_samples_split':
            new_param_dist[param] = [max(2, value - 1), value, value + 1]
        elif param == 'min_samples_leaf':
            new_param_dist[param] = [max(1, value - 1), value, value + 1]
        elif param == 'max_features':
            if value in ['sqrt', 'log2']:
                new_param_dist[param] = [value]
            else:
                new_param_dist[param] = ['sqrt', 'log2', None]
        else:
            new_param_dist[param] = [value]
    return new_param_dist

def train_and_evaluate_model(dataset, best_params_filepath=PARAMS_DIR / 'best_parameters.json'):
    # Prepare features and labels
    features = dataset.drop(['Label'], axis=1)
    labels = dataset['Label']

    # Ensure all features are numeric
    features = features.select_dtypes(include=[np.number])

    # Load existing best parameters if available
    existing_best_params = load_best_parameters(filepath=best_params_filepath)

    if existing_best_params:
        print(f"Loaded existing best parameters for model: {existing_best_params}")
        best_params = existing_best_params
    else:
        # Define initial parameter grid for RandomizedSearchCV
        initial_param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # Perform iterative hyperparameter search
        best_params, best_score = iterative_hyperparameter_search(
            features=features,
            labels=labels,
            initial_param_dist=initial_param_dist,
            max_iterations=5,
            n_iter=20,
            cv=5,
            scoring='f1_weighted',  # Use weighted F1 score for multi-class
            random_state=42
        )

        print(f"Final Best Parameters: {best_params}")
        print(f"Final Best Score: {best_score:.4f}")

        # Save the best parameters
        save_best_parameters({'model': best_params}, filepath=best_params_filepath)

    # Initialize and train the model with the best parameters
    rf_classifier = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')

    # TimeSeriesSplit for walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(features), 1):
        train_features, test_features = features.iloc[train_index], features.iloc[test_index]
        train_labels, test_labels = labels.iloc[train_index], labels.iloc[test_index]

        # Apply SMOTE for balancing classes
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        train_features_resampled, train_labels_resampled = smote.fit_resample(train_features, train_labels)

        # Train the model
        rf_classifier.fit(train_features_resampled, train_labels_resampled)

        # Predict on the test set
        predictions = rf_classifier.predict(test_features)

        # Evaluate the predictions
        precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

    # Calculate average metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    # Save the trained model
    model_filename = MODELS_DIR / 'best_model.pkl'
    joblib.dump(rf_classifier, model_filename)
    print(f"Saved best model as {model_filename}\n")

def main():
    # Load the dataset
    dataset = pd.read_csv(DATA_DIR / 'filtered_dataset.csv')
    
    print(f"Training model for {LABEL_NAME}")
    train_and_evaluate_model(dataset)

if __name__ == "__main__":
    main()