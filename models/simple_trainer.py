#!/usr/bin/env python3
"""
Simple Model Trainer - Creates models without ModelConfig dependency
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_processed_data():
    """Load processed ASSISTments data"""
    try:
        from data.assistments_processor import load_processed_assistments_data
        data_dict = load_processed_assistments_data()
        return data_dict
    except ImportError:
        print("❌ Could not import data processor")
        return None

def create_learner_features(data):
    """Create features for learner classification"""
    features = data.copy()
    
    # Add derived features
    if 'speed_accuracy_tradeoff' not in features.columns:
        features['speed_accuracy_tradeoff'] = features['accuracy'] / features['avg_time_seconds'].replace(0, 1)
    
    if 'persistence' not in features.columns:
        features['persistence'] = features['avg_attempts'] / features['accuracy'].replace(0, 0.1)
    
    # Fill NaN values
    features = features.fillna({
        'consistency': 1.0,
        'speed_accuracy_tradeoff': 0.0,
        'persistence': 1.0,
        'engagement': 0.0,
        'efficiency': 0.5
    })
    
    # Replace infinite values
    features = features.replace([np.inf, -np.inf], 0)
    
    return features

def create_interaction_features(data):
    """Create features for performance prediction"""
    features = data.copy()
    
    # Calculate attempt efficiency
    features['attempt_efficiency'] = 1.0 / features['attempts'].replace(0, 1)
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features

def train_learner_classification_model():
    """Train learner classification model"""
    print("🎯 Training Learner Classification Model...")
    
    # Load data
    data_dict = load_processed_data()
    if not data_dict:
        return None
    
    # Prepare data (sample for prototype)
    learner_profiles = data_dict['learner_profiles'].sample(n=min(500, len(data_dict['learner_profiles'])), random_state=42)
    features = create_learner_features(learner_profiles)
    
    # Select features
    feature_cols = ['accuracy', 'total_questions', 'avg_time_seconds', 
                   'avg_attempts', 'avg_hints_used', 'consistency', 
                   'speed_accuracy_tradeoff', 'persistence', 'engagement', 'efficiency']
    
    X = features[feature_cols]
    y = features['learner_type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'classes': list(y.unique()),
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model_data

def train_performance_prediction_model():
    """Train performance prediction model"""
    print("🎯 Training Performance Prediction Model...")
    
    # Load data
    data_dict = load_processed_data()
    if not data_dict:
        return None
    
    # Prepare data (sample for prototype)
    clean_data = data_dict['clean_data'].sample(n=min(2000, len(data_dict['clean_data'])), random_state=42)
    features = create_interaction_features(clean_data)
    
    # Select features (check what's available)
    available_cols = features.columns.tolist()
    print(f"   Available columns: {available_cols}")
    
    # Use only available features
    feature_cols = ['attempts', 'time_taken_seconds', 'hints_used', 'attempt_efficiency']
    feature_cols = [col for col in feature_cols if col in available_cols]
    
    print(f"   Using features: {feature_cols}")
    
    X = features[feature_cols]
    y = features['correct']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'classes': list(y.unique()),
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model_data

def train_engagement_analysis_model():
    """Train engagement analysis model"""
    print("🎯 Training Engagement Analysis Model...")
    
    # Load data
    data_dict = load_processed_data()
    if not data_dict:
        return None
    
    # Prepare data (sample for prototype)
    clean_data = data_dict['clean_data'].sample(n=min(2000, len(data_dict['clean_data'])), random_state=42)
    
    # Create engagement features
    engagement_data = clean_data.groupby('student_id').agg({
        'correct': ['count', 'mean', 'std'],
        'attempts': 'mean',
        'time_taken_seconds': 'mean'
    }).reset_index()
    
    engagement_data.columns = ['student_id', 'total_interactions', 'avg_accuracy', 'accuracy_std', 'avg_attempts', 'avg_time']
    engagement_data['engagement_level'] = pd.cut(
        engagement_data['total_interactions'],
        bins=[0, 5, 15, float('inf')],
        labels=['low', 'medium', 'high']
    )
    
    engagement_data = engagement_data.sample(n=min(300, len(engagement_data)), random_state=42)
    
    # Select features
    feature_cols = ['total_interactions', 'avg_accuracy', 'accuracy_std', 'avg_attempts', 'avg_time']
    
    X = engagement_data[feature_cols].fillna(0)
    y = engagement_data['engagement_level']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'classes': list(y.unique()),
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model_data

def main():
    """Main training function"""
    print("🚀 Simple Model Training for Adaptive Learning System")
    print("=" * 60)
    
    # Create artifacts directory
    artifacts_dir = "models/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Train models
    models = {}
    
    # 1. Learner Classification
    learner_model = train_learner_classification_model()
    if learner_model:
        models['learner_classification_rf'] = learner_model
    
    # 2. Performance Prediction
    performance_model = train_performance_prediction_model()
    if performance_model:
        models['performance_prediction_gb'] = performance_model
    
    # 3. Engagement Analysis
    engagement_model = train_engagement_analysis_model()
    if engagement_model:
        models['engagement_analysis_rf'] = engagement_model
    
    # Save models
    print("\n💾 Saving models...")
    for model_name, model_data in models.items():
        artifact_path = os.path.join(artifacts_dir, f"{model_name}.pkl")
        
        with open(artifact_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"   ✅ Saved {model_name} -> {artifact_path}")
    
    print(f"\n🎉 Training completed! Saved {len(models)} models")
    print("=" * 60)

if __name__ == "__main__":
    main()
