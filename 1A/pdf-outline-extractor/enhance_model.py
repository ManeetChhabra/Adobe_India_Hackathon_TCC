# enhance_model.py - Additional improvement strategies

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def analyze_errors():
    """Analyze misclassifications to improve model"""
    print("🔍 Error Analysis - Loading current predictions...")
    
    # Load training data
    from train_model import create_labeled_dataset
    
    pdf_files = [f for f in os.listdir("input") if f.endswith(".pdf")]
    dataset = create_labeled_dataset(pdf_files, "input", "ground_truth")
    
    if dataset.empty:
        print("❌ No data available for analysis")
        return
    
    # Prepare features
    features_df = dataset.drop(columns=["text", "page_num", "label"])
    labels = dataset["label"]
    texts = dataset["text"]
    
    # Load trained model
    model = joblib.load("heading_classifier.joblib")
    
    # Get predictions
    predictions = model.predict(features_df)
    probabilities = model.predict_proba(features_df)
    
    # Find misclassifications
    misclassified = []
    for i, (true_label, pred_label) in enumerate(zip(labels, predictions)):
        if true_label != pred_label:
            max_prob = np.max(probabilities[i])
            misclassified.append({
                'text': texts.iloc[i],
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': max_prob,
                'features': features_df.iloc[i].to_dict()
            })
    
    print(f"📊 Analysis Results:")
    print(f"Total samples: {len(dataset)}")
    print(f"Misclassified: {len(misclassified)} ({len(misclassified)/len(dataset)*100:.1f}%)")
    
    # Group by error type
    error_types = {}
    for error in misclassified[:20]:  # Show top 20 errors
        error_key = f"{error['true_label']} → {error['predicted_label']}"
        if error_key not in error_types:
            error_types[error_key] = []
        error_types[error_key].append(error)
    
    print(f"\n🔍 Common Error Patterns:")
    for error_type, errors in error_types.items():
        print(f"{error_type}: {len(errors)} cases")
        for error in errors[:3]:  # Show 3 examples
            print(f"  - '{error['text'][:50]}...' (conf: {error['confidence']:.3f})")

def hyperparameter_tuning():
    """Advanced hyperparameter optimization"""
    print("🎯 Advanced Hyperparameter Tuning...")
    
    from train_model import create_labeled_dataset
    
    pdf_files = [f for f in os.listdir("input") if f.endswith(".pdf")]
    dataset = create_labeled_dataset(pdf_files, "input", "ground_truth")
    
    features_df = dataset.drop(columns=["text", "page_num", "label"])
    labels = dataset["label"]
    
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Advanced parameter grid
    param_grid = {
        'n_estimators': [150, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    print("🔧 Grid search in progress...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='f1_weighted', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation F1: {grid_search.best_score_:.4f}")
    
    # Save optimized model
    joblib.dump(grid_search.best_estimator_, "heading_classifier_optimized.joblib")
    print("✅ Optimized model saved")

def feature_engineering_v2():
    """Create additional engineered features"""
    print("🧪 Advanced Feature Engineering...")
    
    suggestions = """
    🚀 Additional Feature Ideas to Implement:
    
    1. **Sequential Features:**
       - Previous/next line font size ratios
       - Position relative to other headings
       - Heading hierarchy patterns
    
    2. **Document Context:**
       - Distance from document start/end
       - Paragraph break patterns
       - Line spacing consistency
    
    3. **Text Semantic Features:**
       - Common heading words (Introduction, Conclusion, etc.)
       - Capitalization patterns
       - Number/bullet point detection
    
    4. **Layout Features:**
       - Alignment consistency
       - Margin patterns
       - White space analysis
    
    5. **Ensemble Methods:**
       - Combine Mathematical + ML predictions
       - Weighted voting based on confidence
       - Multi-model consensus
    """
    print(suggestions)

if __name__ == "__main__":
    print("🚀 Enhanced Model Analysis & Optimization")
    print("=" * 50)
    
    print("\n1. Error Analysis:")
    analyze_errors()
    
    print("\n2. Feature Engineering Suggestions:")
    feature_engineering_v2()
    
    print("\n3. Hyperparameter Tuning (Optional - takes time):")
    choice = input("Run hyperparameter optimization? (y/N): ")
    if choice.lower() == 'y':
        hyperparameter_tuning()
