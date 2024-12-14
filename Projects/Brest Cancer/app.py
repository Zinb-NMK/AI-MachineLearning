import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from flask import Flask, render_template, send_file, jsonify
import io

# Create Flask app
app = Flask(__name__)

# Load dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Feature categorization
features_mean = list(data.columns[1:11])
features_se = list(data.columns[11:20])
features_worst = list(data.columns[21:31])


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route to show diagnosis count plot
@app.route('/diagnosis_count_plot')
def diagnosis_count_plot():
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diagnosis', data=data, palette='coolwarm', hue='diagnosis')
    plt.title("Diagnosis Count")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


# Route to show correlation heatmap
@app.route('/correlation_heatmap')
def correlation_heatmap():
    plt.figure(figsize=(14, 14))
    corr = data[features_mean].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap (Mean Features)")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


# Route to evaluate Random Forest Classifier
@app.route('/evaluate_rf')
def evaluate_rf():
    prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    target_var = 'diagnosis'
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    train_X = train[prediction_var]
    train_y = train[target_var]
    test_X = test[prediction_var]
    test_y = test[target_var]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(train_X, train_y)
    rf_predictions = rf_model.predict(test_X)
    rf_accuracy = metrics.accuracy_score(test_y, rf_predictions)

    return jsonify({'model': 'Random Forest', 'accuracy': f"{rf_accuracy:.3%}"})


# Route to evaluate multiple models
@app.route('/evaluate_models')
def evaluate_models():
    prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
    target_var = 'diagnosis'
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    train_X = train[prediction_var]
    train_y = train[target_var]

    models = [
        LogisticRegression(random_state=42, max_iter=1000),
        DecisionTreeClassifier(random_state=42),
        KNeighborsClassifier(),
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(random_state=42)
    ]

    results = []
    for model in models:
        model_name = model.__class__.__name__
        model.fit(train_X, train_y)
        accuracy = model.score(test[prediction_var], test[target_var])
        results.append({'model': model_name, 'accuracy': f"{accuracy:.3%}"})

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
