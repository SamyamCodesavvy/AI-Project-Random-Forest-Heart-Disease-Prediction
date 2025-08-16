from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from decisiontree import DecisionTree
from randomforest import RandomForest
import matplotlib
import pandas as pd
from sklearn import datasets
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import math

app = Flask(__name__)

# Initialize global variable for trained model
trained_model = None

# Load dataset
df = pd.read_csv("heart.csv")
column_names = df.drop(columns=['target']).columns.tolist()

# Separate features (X) and target (y)
X = df.drop(columns=['target']).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def convert_max_features(max_features_str, n_features):
    """Convert string max_features to appropriate value"""
    if max_features_str == "None":
        return None
    elif max_features_str == "auto" or max_features_str == "sqrt":
        result = int(math.sqrt(n_features))
        return max(1, min(result, n_features))  # FIXED: Ensure valid range
    elif max_features_str == "log2":
        result = int(math.log2(n_features)) if n_features > 1 else 1
        return max(1, min(result, n_features))  # FIXED: Ensure valid range
    else:
        try:
            result = int(max_features_str)
            return max(1, min(result, n_features))  # FIXED: Ensure valid range
        except ValueError:
            return None

@app.route('/')
def index():
    return render_template('index.html', column_names=column_names)

@app.route('/run_rf', methods=['POST'])
def run_rf():
    global trained_model
    
    try:
        time.sleep(0.5)
        
        # Get parameters from frontend
        n_estimators = int(request.form.get('n_estimators', 10))
        max_depth = int(request.form.get('max_depth', 15))
        max_features_str = request.form.get('max_features', 'None')
        bootstrap = request.form.get('bootstrap', 'True') == 'True'
        max_samples = int(request.form.get('max_samples', len(X_train)))
        
        # Validate parameters
        if n_estimators < 1 or n_estimators > 200:
            return jsonify({'success': False, 'error': 'Number of estimators must be between 1 and 200'}), 400
        if max_depth < 1 or max_depth > 50:
            return jsonify({'success': False, 'error': 'Max depth must be between 1 and 50'}), 400
        if max_samples < 1:
            return jsonify({'success': False, 'error': 'Max samples must be positive'}), 400
        
        # Convert max_features string to appropriate value
        n_features = X_train.shape[1]
        max_features = convert_max_features(max_features_str, n_features)
        
        # Validate and adjust max_samples
        max_samples = max(1, min(max_samples, len(X_train)))
        
        # Create training data
        if bootstrap and max_samples < len(X_train):
            X_train_sample, y_train_sample = create_bootstrap_samples(X_train, y_train, max_samples)
        else:
            X_train_sample, y_train_sample = X_train[:max_samples], y_train[:max_samples]
        
        # FIXED: Ensure we have valid training data
        if len(X_train_sample) == 0 or len(y_train_sample) == 0:
            return jsonify({'success': False, 'error': 'No valid training samples'}), 400
        
        # Create and train Random Forest
        clf = RandomForest(
            n_trees=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            n_features=max_features,
            random_state=42
        )
        
        # Fit the model
        clf.fit(X_train_sample, y_train_sample)
        
        # FIXED: Check if any trees were trained
        if len(clf.trees) == 0:
            return jsonify({'success': False, 'error': 'No trees were successfully trained'}), 400
        
        # Make predictions
        y_pred_train = clf.predict(X_train_sample)
        y_pred_test = clf.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train_sample, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        # Generate plots
        decision_boundary_plot = plot_decision_boundary(clf, X_test, y_test, feature_indices=(0, 1))
        confusion_matrix_plot = plot_confusion_matrix(y_test, y_pred_test)
        
        # Store the trained model
        trained_model = clf
        
        response = {
            'success': True,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'plots': {
                'decision_boundary': decision_boundary_plot,
                'confusion_matrix': confusion_matrix_plot
            },
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features_str,
                'bootstrap': bootstrap,
                'max_samples': max_samples,
                'actual_training_samples': len(X_train_sample),
                'actual_max_features': max_features,
                'trees_trained': len(clf.trees)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Log full error for debugging
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Rest of the functions remain the same...
def plot_decision_boundary(clf, X, y, feature_indices=(0, 1)):
    """Create decision boundary plot for specified feature indices"""
    try:
        # Ensure valid feature indices
        if len(feature_indices) != 2 or feature_indices[0] == feature_indices[1]:
            raise ValueError("Must provide two distinct feature indices")
        if max(feature_indices) >= X.shape[1]:
            raise ValueError("Feature indices out of bounds")
        
        # Create a mesh grid for the two selected features
        x_min, x_max = X[:, feature_indices[0]].min() - 1, X[:, feature_indices[0]].max() + 1
        y_min, y_max = X[:, feature_indices[1]].min() - 1, X[:, feature_indices[1]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Create input for prediction
        n_features = X.shape[1]
        X_grid = np.zeros((xx.ravel().shape[0], n_features))
        for i in range(n_features):
            if i == feature_indices[0]:
                X_grid[:, i] = xx.ravel()
            elif i == feature_indices[1]:
                X_grid[:, i] = yy.ravel()
            else:
                X_grid[:, i] = np.mean(X[:, i])
        
        # Predict on the grid
        Z = clf.predict(X_grid)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        scatter = plt.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], c=y, 
                             cmap='coolwarm', edgecolor='k', s=50)
        plt.colorbar(scatter)
        plt.xlabel(column_names[feature_indices[0]])
        plt.ylabel(column_names[feature_indices[1]])
        plt.title(f'Decision Boundary ({column_names[feature_indices[0]]} vs {column_names[feature_indices[1]]})')
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return img_base64
    except Exception as e:
        print(f"Error in plot_decision_boundary: {e}")
        return ""

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix heatmap"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Disease', 'Heart Disease'],
                    yticklabels=['No Disease', 'Heart Disease'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return img_base64
    except Exception as e:
        print(f"Error in plot_confusion_matrix: {e}")
        return ""

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': len(y_true),
            'class_report': report
        }
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'support': len(y_true),
            'class_report': {}
        }

def create_bootstrap_samples(X, y, max_samples):
    """Create bootstrap samples with specified max_samples"""
    n_samples = min(max_samples, len(X))
    n_samples = max(1, n_samples)
    indices = np.random.choice(len(X), n_samples, replace=True)
    return X[indices], y[indices]

@app.route('/plot_decision_boundary', methods=['POST'])
def plot_decision_boundary_route():
    global trained_model
    
    try:
        feature1 = int(request.form.get('feature1'))
        feature2 = int(request.form.get('feature2'))
        
        if feature1 == feature2:
            return jsonify({'success': False, 'error': 'Features must be distinct'}), 400
        if feature1 < 0 or feature2 < 0 or feature1 >= len(column_names) or feature2 >= len(column_names):
            return jsonify({'success': False, 'error': 'Invalid feature indices'}), 400
        
        if trained_model is None:
            return jsonify({'success': False, 'error': 'No trained model available. Run the algorithm first.'}), 400
        
        decision_boundary_plot = plot_decision_boundary(trained_model, X_test, y_test, 
                                                      feature_indices=(feature1, feature2))
        
        return jsonify({
            'success': True,
            'plot': decision_boundary_plot
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

# ============================================================================
