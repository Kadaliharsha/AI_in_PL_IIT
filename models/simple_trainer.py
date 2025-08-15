#!/usr/bin/env python3
"""
Simple Model Trainer - Creates models without ModelConfig dependency
Enhanced with comprehensive metrics and debugging
"""

import os
import sys
import pickle
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    f1_score, roc_auc_score, precision_score, recall_score
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_features(features, name="features"):
    """Debug feature quality and statistics"""
    print(f"🔍 Debugging {name}:")
    print(f"   Shape: {features.shape}")
    print(f"   Data types: {features.dtypes.value_counts().to_dict()}")
    print(f"   NaN count: {features.isnull().sum().sum()}")
    
    # Check for infinite values in numeric columns
    numeric_features = features.select_dtypes(include=[np.number])
    if not numeric_features.empty:
        inf_count = np.isinf(numeric_features).sum().sum()
        print(f"   Inf count: {inf_count}")
    
    print(f"   Value ranges:")
    print(f"     {features.describe().round(3)}")
    print(f"   Sample values:")
    print(f"     {features.head(3).to_string()}")
    print("-" * 50)

def debug_model_performance(y_true, y_pred, model_name):
    """Debug model predictions and errors"""
    print(f"🎯 {model_name} Performance Debug:")
    print(f"   True classes: {np.unique(y_true)}")
    print(f"   Predicted classes: {np.unique(y_pred)}")
    
    # Handle both numeric and string labels
    try:
        # Try to convert to numeric for bincount
        y_true_numeric = pd.Categorical(y_true).codes
        y_pred_numeric = pd.Categorical(y_pred).codes
        print(f"   Class distribution (true): {np.bincount(y_true_numeric)}")
        print(f"   Class distribution (pred): {np.bincount(y_pred_numeric)}")
    except Exception as e:
        # Fallback for string labels
        if hasattr(y_true, 'value_counts'):
            print(f"   Class distribution (true): {y_true.value_counts().to_dict()}")
            print(f"   Class distribution (pred): {y_pred.value_counts().to_dict()}")
        else:
            # Handle numpy arrays
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            print(f"   Class distribution (true): {dict(zip(unique_true, counts_true))}")
            print(f"   Class distribution (pred): {dict(zip(unique_pred, counts_pred))}")
    
    # Show some misclassifications
    errors = y_true != y_pred
    if errors.any():
        print(f"   Misclassifications: {errors.sum()}/{len(y_true)} ({errors.sum()/len(y_true)*100:.1f}%)")
        if len(y_true) > 0:
            error_indices = np.where(errors)[0][:3]  # First 3 errors
            # Convert to list safely
            try:
                true_errors = [str(y_true[i]) for i in error_indices]
                pred_errors = [str(y_pred[i]) for i in error_indices]
                print(f"   Error examples: True={true_errors}, Pred={pred_errors}")
            except Exception as e:
                print(f"   Error examples: Could not extract (error: {e})")
    print("-" * 50)

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, model_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    print(f"📊 Calculating comprehensive metrics for {model_name}...")
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # F1 scores
    try:
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted'))
        metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro'))
    except Exception as e:
        print(f"   ⚠️ F1 score calculation failed: {e}")
        metrics['f1_macro'] = 0.0
        metrics['f1_weighted'] = 0.0
        metrics['f1_micro'] = 0.0
    
    # Precision and Recall
    try:
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    except Exception as e:
        print(f"   ⚠️ Precision/Recall calculation failed: {e}")
        metrics['precision_macro'] = 0.0
        metrics['recall_macro'] = 0.0
    
    # ROC-AUC (only for binary classification)
    if len(np.unique(y_true)) == 2 and y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            print(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            print(f"   ⚠️ ROC-AUC calculation failed: {e}")
            metrics['roc_auc'] = 0.0
    else:
        print(f"   ℹ️ ROC-AUC: Not applicable (multiclass or no probabilities)")
        metrics['roc_auc'] = None
    
    # Confusion Matrix
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        print(f"   ✅ Confusion Matrix shape: {conf_matrix.shape}")
    except Exception as e:
        print(f"   ⚠️ Confusion matrix calculation failed: {e}")
        metrics['confusion_matrix'] = []
    
    # Classification Report
    try:
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = class_report
        print(f"   ✅ Classification Report generated")
    except Exception as e:
        print(f"   ⚠️ Classification report failed: {e}")
        metrics['classification_report'] = {}
    
    print(f"   📈 Final Metrics Summary:")
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"     Precision-Macro: {metrics['precision_macro']:.4f}")
    print(f"     Recall-Macro: {metrics['recall_macro']:.4f}")
    
    return metrics

def load_processed_data():
    """Load processed ASSISTments data"""
    try:
        # Try to import from data processor
        from data.assistments_processor import load_processed_assistments_data
        data_dict = load_processed_assistments_data()
        return data_dict
    except ImportError:
        # Fallback: try direct file loading
        try:
            import pandas as pd
            import os
            
            # Check if processed files exist
            processed_dir = "data/processed"
            if not os.path.exists(processed_dir):
                print(f"❌ Processed data directory not found: {processed_dir}")
                return None
            
            # Load files directly
            clean_data_path = os.path.join(processed_dir, "clean_assistments_data.csv")
            profiles_path = os.path.join(processed_dir, "learner_profiles.csv")
            question_bank_path = os.path.join(processed_dir, "question_bank.csv")
            
            if not all(os.path.exists(p) for p in [clean_data_path, profiles_path, question_bank_path]):
                print("❌ Some processed data files are missing")
                print(f"   Required: {clean_data_path}, {profiles_path}, {question_bank_path}")
                return None
            
            print("📁 Loading processed data directly...")
            clean_data = pd.read_csv(clean_data_path)
            profiles = pd.read_csv(profiles_path)
            question_bank = pd.read_csv(question_bank_path)
            
            return {
                'clean_data': clean_data,
                'learner_profiles': profiles,
                'question_bank': question_bank
            }
            
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
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
    start_time = time.time()
    
    # Load data
    data_dict = load_processed_data()
    if not data_dict:
        print("❌ Failed to load data")
        return None
    
    # Prepare data (sample for prototype)
    learner_profiles = data_dict['learner_profiles'].sample(n=min(500, len(data_dict['learner_profiles'])), random_state=42)
    print(f"   📊 Using {len(learner_profiles)} learner profiles")
    
    features = create_learner_features(learner_profiles)
    
    # Debug features
    debug_features(features, "Learner Classification Features")
    
    # Select features
    feature_cols = ['accuracy', 'total_questions', 'avg_time_seconds', 
                   'avg_attempts', 'avg_hints_used', 'consistency', 
                   'speed_accuracy_tradeoff', 'persistence', 'engagement', 'efficiency']
    
    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in features.columns]
    if missing_features:
        print(f"   ⚠️ Missing features: {missing_features}")
        feature_cols = [col for col in feature_cols if col in features.columns]
        print(f"   ✅ Using available features: {feature_cols}")
    
    X = features[feature_cols]
    y = features['learner_type']
    
    print(f"   🎯 Target classes: {y.unique()}")
    print(f"   📊 Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   📈 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])
    
    # Train model
    print("   🚀 Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Debug performance
    debug_model_performance(y_test, y_pred, "Learner Classification")
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba, "Learner Classification")
    
    # Cross-validation
    print("   🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    metrics['cv_mean'] = float(cv_scores.mean())
    metrics['cv_std'] = float(cv_scores.std())
    
    print(f"   📊 Cross-Validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
    
    # Training time
    training_time = time.time() - start_time
    metrics['training_time_seconds'] = training_time
    print(f"   ⏱️ Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'classes': list(y.unique()),
        'metrics': metrics,
        'training_info': {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'random_state': 42,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    return model_data

def train_performance_prediction_model():
    """Train performance prediction model"""
    print("🎯 Training Performance Prediction Model...")
    start_time = time.time()
    
    # Load data
    data_dict = load_processed_data()
    if not data_dict:
        print("❌ Failed to load data")
        return None
    
    # Prepare data (sample for prototype)
    clean_data = data_dict['clean_data'].sample(n=min(2000, len(data_dict['clean_data'])), random_state=42)
    print(f"   📊 Using {len(clean_data)} interaction records")
    
    features = create_interaction_features(clean_data)
    
    # Debug features
    debug_features(features, "Performance Prediction Features")
    
    # Select features (check what's available)
    available_cols = features.columns.tolist()
    print(f"   🔍 Available columns: {available_cols}")
    
    # Use only available features
    feature_cols = ['attempts', 'time_taken_seconds', 'hints_used', 'attempt_efficiency']
    feature_cols = [col for col in feature_cols if col in available_cols]
    
    print(f"   ✅ Using features: {feature_cols}")
    
    X = features[feature_cols]
    y = features['correct']
    
    print(f"   🎯 Target classes: {y.unique()}")
    print(f"   📊 Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   📈 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    # Train model
    print("   🚀 Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Debug performance
    debug_model_performance(y_test, y_pred, "Performance Prediction")
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba, "Performance Prediction")
    
    # Cross-validation
    print("   🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    metrics['cv_mean'] = float(cv_scores.mean())
    metrics['cv_std'] = float(cv_scores.std())
    
    print(f"   📊 Cross-Validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
    
    # Training time
    training_time = time.time() - start_time
    metrics['training_time_seconds'] = training_time
    print(f"   ⏱️ Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'classes': list(y.unique()),
        'metrics': metrics,
        'training_info': {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'random_state': 42,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    return model_data

def train_engagement_analysis_model():
    """Train engagement analysis model with balanced data"""
    print("🎯 Training Engagement Analysis Model with Balanced Data...")
    start_time = time.time()
    
    # Load data
    data_dict = load_processed_data()
    if not data_dict:
        print("❌ Failed to load data")
        return None
    
    # Prepare data (sample for prototype)
    clean_data = data_dict['clean_data'].sample(n=min(2000, len(data_dict['clean_data'])), random_state=42)
    print(f"   📊 Using {len(clean_data)} interaction records")
    
    # Create engagement features
    engagement_data = clean_data.groupby('student_id').agg({
        'correct': ['count', 'mean', 'std'],
        'attempts': 'mean',
        'time_taken_seconds': 'mean'
    }).reset_index()
    
    engagement_data.columns = ['student_id', 'total_interactions', 'avg_accuracy', 'accuracy_std', 'avg_attempts', 'avg_time']
    
    # Create balanced engagement levels based on performance quality
    # Calculate engagement score (0-100) based on multiple factors
    engagement_data['engagement_score'] = (
        engagement_data['avg_accuracy'] * 40 +  # Accuracy weight: 40 points
        (1 - engagement_data['avg_attempts'] / 3) * 30 +  # Efficiency weight: 30 points
        (1 - engagement_data['avg_time'] / 120) * 30  # Speed weight: 30 points
    ).clip(0, 100)
    
    # Create balanced engagement levels
    engagement_data['engagement_level'] = pd.cut(
        engagement_data['engagement_score'],
        bins=[0, 40, 70, 100],
        labels=['low', 'medium', 'high']
    )
    
    # Debug engagement data
    print(f"   🎯 Engagement levels created:")
    print(f"     {engagement_data['engagement_level'].value_counts().to_dict()}")
    
    # Ensure balanced dataset by sampling equal numbers from each class
    low_samples = engagement_data[engagement_data['engagement_level'] == 'low']
    medium_samples = engagement_data[engagement_data['engagement_level'] == 'medium']
    high_samples = engagement_data[engagement_data['engagement_level'] == 'high']
    
    # Find the minimum class size
    min_class_size = min(len(low_samples), len(medium_samples), len(high_samples))
    print(f"   📊 Minimum class size: {min_class_size}")
    
    # Sample equal numbers from each class for balanced training
    balanced_data = pd.concat([
        low_samples.sample(n=min(min_class_size, 100), random_state=42),
        medium_samples.sample(n=min(min_class_size, 100), random_state=42),
        high_samples.sample(n=min(min_class_size, 100), random_state=42)
    ])
    
    print(f"   📊 Balanced dataset size: {len(balanced_data)}")
    print(f"   📊 Balanced class distribution: {balanced_data['engagement_level'].value_counts().to_dict()}")
    
    # Select features
    feature_cols = ['total_interactions', 'avg_accuracy', 'accuracy_std', 'avg_attempts', 'avg_time']
    
    X = balanced_data[feature_cols].fillna(0)
    y = balanced_data['engagement_level']
    
    # Debug features
    debug_features(X, "Engagement Analysis Features")
    
    print(f"   🎯 Target classes: {y.unique()}")
    print(f"   📊 Final class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   📈 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'))
    ])
    
    # Train model
    print("   🚀 Training pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    # Debug performance
    debug_model_performance(y_test, y_pred, "Engagement Analysis")
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba, "Engagement Analysis")
    
    # Cross-validation
    print("   🔄 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
    metrics['cv_mean'] = float(cv_scores.mean())
    metrics['cv_std'] = float(cv_scores.std())
    
    print(f"   📊 Cross-Validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
    
    # Training time
    training_time = time.time() - start_time
    metrics['training_time_seconds'] = training_time
    print(f"   ⏱️ Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'classes': list(y.unique()),
        'metrics': metrics,
        'training_info': {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'random_state': 42,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    return model_data

def main():
    """Main training function"""
    print("🚀 Simple Model Training for Adaptive Learning System")
    print("Enhanced with Comprehensive Metrics and Debugging")
    print("=" * 70)
    
    overall_start_time = time.time()
    
    # Create artifacts directory
    artifacts_dir = "models/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Train models
    models = {}
    training_summary = {
        'training_session': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_models': 3,
            'successful_models': 0,
            'failed_models': 0
        },
        'models': {}
    }
    
    print("\n🎯 Starting Model Training...")
    print("=" * 50)
    
    # 1. Learner Classification
    print("\n1️⃣ Training Learner Classification Model...")
    learner_model = train_learner_classification_model()
    if learner_model:
        models['learner_classification_rf'] = learner_model
        training_summary['models']['learner_classification_rf'] = {
            'status': 'success',
            'metrics': learner_model['metrics'],
            'training_info': learner_model['training_info']
        }
        training_summary['training_session']['successful_models'] += 1
        print("   ✅ Learner Classification Model trained successfully!")
    else:
        training_summary['models']['learner_classification_rf'] = {'status': 'failed'}
        training_summary['training_session']['failed_models'] += 1
        print("   ❌ Learner Classification Model training failed!")
    
    # 2. Performance Prediction
    print("\n2️⃣ Training Performance Prediction Model...")
    performance_model = train_performance_prediction_model()
    if performance_model:
        models['performance_prediction_gb'] = performance_model
        training_summary['models']['performance_prediction_gb'] = {
            'status': 'success',
            'metrics': performance_model['metrics'],
            'training_info': performance_model['training_info']
        }
        training_summary['training_session']['successful_models'] += 1
        print("   ✅ Performance Prediction Model trained successfully!")
    else:
        training_summary['models']['performance_prediction_gb'] = {'status': 'failed'}
        training_summary['training_session']['failed_models'] += 1
        print("   ❌ Performance Prediction Model training failed!")
    
    # 3. Engagement Analysis
    print("\n3️⃣ Training Engagement Analysis Model...")
    engagement_model = train_engagement_analysis_model()
    if engagement_model:
        models['engagement_analysis_rf'] = engagement_model
        training_summary['models']['engagement_analysis_rf'] = {
            'status': 'success',
            'metrics': engagement_model['metrics'],
            'training_info': engagement_model['training_info']
        }
        training_summary['training_session']['successful_models'] += 1
        print("   ✅ Engagement Analysis Model trained successfully!")
    else:
        training_summary['models']['engagement_analysis_rf'] = {'status': 'failed'}
        training_summary['training_session']['failed_models'] += 1
        print("   ❌ Engagement Analysis Model training failed!")
    
    # Save models and metrics
    print(f"\n💾 Saving Models and Metrics...")
    print("=" * 50)
    
    for model_name, model_data in models.items():
        # Save model pickle
        artifact_path = os.path.join(artifacts_dir, f"{model_name}.pkl")
        with open(artifact_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"   ✅ Saved {model_name} -> {artifact_path}")
        
        # Save metrics JSON
        metrics_path = os.path.join(artifacts_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(model_data['metrics'], f, indent=2)
        print(f"   📊 Saved metrics -> {metrics_path}")
    
    # Save overall training summary
    overall_training_time = time.time() - overall_start_time
    training_summary['training_session']['total_training_time_seconds'] = overall_training_time
    
    summary_path = os.path.join(artifacts_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"   📋 Saved training summary -> {summary_path}")
    
    # Final summary
    print(f"\n🎉 Training Session Completed!")
    print("=" * 50)
    print(f"   📊 Total Models: {training_summary['training_session']['total_models']}")
    print(f"   ✅ Successful: {training_summary['training_session']['successful_models']}")
    print(f"   ❌ Failed: {training_summary['training_session']['failed_models']}")
    print(f"   ⏱️ Total Time: {overall_training_time:.2f} seconds")
    
    if training_summary['training_session']['successful_models'] > 0:
        print(f"\n🏆 Model Performance Summary:")
        print("-" * 30)
        for model_name, model_info in training_summary['models'].items():
            if model_info['status'] == 'success':
                metrics = model_info['metrics']
                print(f"   {model_name}:")
                print(f"     Accuracy: {metrics['accuracy']:.4f}")
                print(f"     F1-Macro: {metrics['f1_macro']:.4f}")
                print(f"     CV Score: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
                print(f"     Training Time: {metrics['training_time_seconds']:.2f}s")
    
    print(f"\n📁 All artifacts saved to: {artifacts_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
