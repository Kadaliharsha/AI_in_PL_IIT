#!/usr/bin/env python3
"""
AI Model Integration for Adaptive Learning System
Enhanced with metrics access and debugging capabilities
"""

import os
import pickle
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

class AIModelManager:
    """Manages AI models for adaptive learning recommendations"""
    
    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.artifacts_dir = artifacts_dir
        self.models = {}
        self.metrics = {}
        self.training_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models with enhanced error handling"""
        print("🔄 Loading AI Models...")
        
        model_files = [
            "learner_classification_rf.pkl", 
            "performance_prediction_gb.pkl",
            "engagement_analysis_rf.pkl"
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.artifacts_dir, model_file)
            model_name = model_file.replace('.pkl', '')
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Extract model components
                    if isinstance(model_data, dict) and 'pipeline' in model_data:
                        self.models[model_name] = {
                            'pipeline': model_data['pipeline'],
                            'feature_names': model_data.get('feature_names', []),
                            'classes': model_data.get('classes', []),
                            'metrics': model_data.get('metrics', {}),
                            'training_info': model_data.get('training_info', {})
                        }
                        
                        # Store metrics separately for easy access
                        self.metrics[model_name] = model_data.get('metrics', {})
                        self.training_info[model_name] = model_data.get('training_info', {})
                        
                        print(f"✅ Loaded {model_name}")
                        print(f"   📊 Metrics: Accuracy={self.metrics[model_name].get('accuracy', 'N/A'):.4f}, "
                              f"F1={self.metrics[model_name].get('f1_macro', 'N/A'):.4f}")
                        
                    else:
                        print(f"⚠️ Invalid model format for {model_file}")
                        
                except Exception as e:
                    print(f"❌ Error loading {model_file}: {e}")
                    # Try alternative loading method
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f, encoding='latin1')
                            if isinstance(model_data, dict) and 'pipeline' in model_data:
                                self.models[model_name] = {
                                    'pipeline': model_data['pipeline'],
                                    'feature_names': model_data.get('feature_names', []),
                                    'classes': model_data.get('classes', []),
                                    'metrics': model_data.get('metrics', {}),
                                    'training_info': model_data.get('training_info', {})
                                }
                                self.metrics[model_name] = model_data.get('metrics', {})
                                self.training_info[model_name] = model_data.get('training_info', {})
                                print(f"✅ Loaded {model_name} (alternative method)")
                    except:
                        print(f"❌ Failed to load {model_file} with alternative method")
            else:
                print(f"⚠️ Model file not found: {model_file}")
        
        print(f"🎯 Loaded {len(self.models)} models successfully")
        print("-" * 50)
    
    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a specific model"""
        return self.metrics.get(model_name, {})
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all loaded models"""
        return self.metrics
    
    def get_training_info(self, model_name: str) -> Dict[str, Any]:
        """Get training information for a specific model"""
        return self.training_info.get(model_name, {})
    
    def debug_prediction_input(self, features: Dict[str, Any], model_name: str):
        """Debug the input features for prediction"""
        print(f"🔍 Debugging prediction input for {model_name}:")
        print(f"   Input features: {features}")
        
        if model_name in self.models:
            expected_features = self.models[model_name]['feature_names']
            print(f"   Expected features: {expected_features}")
            
            missing_features = [f for f in expected_features if f not in features]
            if missing_features:
                print(f"   ⚠️ Missing features: {missing_features}")
            
            extra_features = [f for f in features if f not in expected_features]
            if extra_features:
                print(f"   ℹ️ Extra features: {extra_features}")
        print("-" * 50)
    
    def predict_learner_type(self, student_features: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Predict learner type with enhanced debugging"""
        model_name = 'learner_classification_rf'
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Debug input
        self.debug_prediction_input(student_features, model_name)
        
        # Prepare features
        feature_cols = self.models[model_name]['feature_names']
        X = pd.DataFrame([[student_features.get(col, 0) for col in feature_cols]], columns=feature_cols)
        
        # Make prediction with timing
        start_time = time.time()
        try:
            prediction = self.models[model_name]['pipeline'].predict(X)[0]
            prediction_proba = self.models[model_name]['pipeline'].predict_proba(X)[0]
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Get class labels first
            classes = self.models[model_name]['classes']
            
            # Debug: Show what the model returned
            print(f"🔍 Debug: Model returned prediction={prediction} (type: {type(prediction)})")
            print(f"🔍 Debug: Classes available: {classes}")
            
            # Get confidence (max probability)
            confidence = float(np.max(prediction_proba))
            
            # Handle both string and numeric predictions
            if isinstance(prediction, str):
                # If prediction is already a string (class name), use it directly
                predicted_class = prediction
            elif isinstance(prediction, (int, np.integer)):
                # If prediction is numeric index, convert to class name
                predicted_class = classes[prediction] if prediction < len(classes) else "unknown"
            else:
                # Fallback for unexpected types
                predicted_class = str(prediction)
            
            print(f"🔍 Debug: Final predicted_class: {predicted_class}")
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': dict(zip(classes, prediction_proba.tolist())),
                'latency_ms': latency,
                'model_metrics': self.metrics[model_name]
            }
            
            print(f"✅ {model_name} prediction: {predicted_class} (confidence: {confidence:.3f}, latency: {latency:.2f}ms)")
            
            return predicted_class, confidence, result
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            print(f"❌ {model_name} prediction failed: {e} (latency: {latency:.2f}ms)")
            raise
    
    def predict_performance(self, question_features: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Predict performance with enhanced debugging"""
        model_name = 'performance_prediction_gb'
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Debug input
        self.debug_prediction_input(question_features, model_name)
        
        # Prepare features
        feature_cols = self.models[model_name]['feature_names']
        X = pd.DataFrame([[question_features.get(col, 0) for col in feature_cols]], columns=feature_cols)
        
        # Make prediction with timing
        start_time = time.time()
        try:
            prediction_proba = self.models[model_name]['pipeline'].predict_proba(X)[0]
            latency = (time.time() - start_time) * 1000
            
            # Get success probability (class 1)
            success_prob = float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.0
            
            result = {
                'success_probability': success_prob,
                'failure_probability': float(prediction_proba[0]) if len(prediction_proba) > 1 else 0.0,
                'latency_ms': latency,
                'model_metrics': self.metrics[model_name]
            }
            
            print(f"✅ {model_name} prediction: success_prob={success_prob:.3f}, latency={latency:.2f}ms")
            
            return success_prob, 1 - success_prob, result
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            print(f"❌ {model_name} prediction failed: {e} (latency: {latency:.2f}ms)")
            raise
    
    def analyze_engagement(self, behavior_features: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Analyze engagement with enhanced debugging"""
        model_name = 'engagement_analysis_rf'
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Debug input
        self.debug_prediction_input(behavior_features, model_name)
        
        # Prepare features
        feature_cols = self.models[model_name]['feature_names']
        X = pd.DataFrame([[behavior_features.get(col, 0) for col in feature_cols]], columns=feature_cols)
        
        # Make prediction with timing
        start_time = time.time()
        try:
            prediction = self.models[model_name]['pipeline'].predict(X)[0]
            prediction_proba = self.models[model_name]['pipeline'].predict_proba(X)[0]
            latency = (time.time() - start_time) * 1000
            
            # Get class labels
            classes = self.models[model_name]['classes']
            
            # Handle both string and numeric predictions
            if isinstance(prediction, str):
                # If prediction is already a string (class name), use it directly
                predicted_level = prediction
            elif isinstance(prediction, (int, np.integer)):
                # If prediction is numeric index, convert to class name
                predicted_level = classes[prediction] if prediction < len(classes) else "unknown"
            else:
                # Fallback for unexpected types
                predicted_level = str(prediction)
            
            # Get confidence (max probability)
            confidence = float(np.max(prediction_proba))
            
            result = {
                'engagement_level': predicted_level,
                'confidence': confidence,
                'level_probabilities': dict(zip(classes, prediction_proba.tolist())),
                'latency_ms': latency,
                'model_metrics': self.metrics[model_name]
            }
            
            print(f"✅ {model_name} prediction: {predicted_level} (confidence: {confidence:.3f}, latency: {latency:.2f}ms)")
            
            return predicted_level, confidence, result
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            print(f"❌ {model_name} prediction failed: {e} (latency: {latency:.2f}ms)")
            raise
    
    def get_adaptive_recommendations(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive adaptive recommendations with enhanced debugging"""
        print("🎯 Generating adaptive recommendations...")
        start_time = time.time()
        
        try:
            # Predict learner type
            learner_type, learner_confidence, learner_result = self.predict_learner_type(student_data)
            
            # Analyze engagement
            engagement_level, engagement_confidence, engagement_result = self.analyze_engagement(student_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(learner_type, engagement_level, student_data)
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            
            result = {
                'learner_type': learner_type,
                'learner_confidence': learner_confidence,
                'engagement_level': engagement_level,
                'engagement_confidence': engagement_confidence,
                'recommendations': recommendations,
                'total_latency_ms': total_latency,
                'model_details': {
                    'learner_model': learner_result,
                    'engagement_model': engagement_result
                }
            }
            
            print(f"✅ Recommendations generated in {total_latency:.2f}ms")
            print(f"   Learner: {learner_type} (confidence: {learner_confidence:.3f})")
            print(f"   Engagement: {engagement_level} (confidence: {engagement_confidence:.3f})")
            
            return result
            
        except Exception as e:
            total_latency = (time.time() - start_time) * 1000
            print(f"❌ Recommendation generation failed: {e} (latency: {total_latency:.2f}ms)")
            raise
    
    def _generate_recommendations(self, learner_type: str, engagement_level: str, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized recommendations based on predictions"""
        print(f"🔍 Debug: learner_type='{learner_type}' (type: {type(learner_type)})")
        print(f"🔍 Debug: engagement_level='{engagement_level}' (type: {type(engagement_level)})")
        
        recommendations = {
            'study_plan': [],
            'difficulty_adjustment': '',
            'motivation_tips': [],
            'resources': [],
            'next_steps': []
        }
        
        # Study plan based on learner type
        if learner_type == 'struggling':
            recommendations['study_plan'] = [
                "Focus on foundational concepts",
                "Practice with easier problems first",
                "Use hints and explanations",
                "Review previous material regularly"
            ]
            recommendations['difficulty_adjustment'] = "Start with easier problems and gradually increase difficulty"
        elif learner_type == 'average':
            recommendations['study_plan'] = [
                "Balance practice and new concepts",
                "Mix easy and challenging problems",
                "Focus on problem-solving strategies",
                "Regular review and practice"
            ]
            recommendations['difficulty_adjustment'] = "Maintain current difficulty with occasional challenges"
        elif learner_type == 'advanced':
            recommendations['study_plan'] = [
                "Tackle challenging problems",
                "Explore advanced concepts",
                "Help peers with difficult problems",
                "Focus on efficiency and speed"
            ]
            recommendations['difficulty_adjustment'] = "Increase difficulty and introduce advanced topics"
        else:
            # Default case
            recommendations['study_plan'] = [
                "Focus on core concepts",
                "Practice regularly",
                "Seek help when needed",
                "Track your progress"
            ]
            recommendations['difficulty_adjustment'] = "Maintain balanced difficulty"
        
        # Motivation tips based on engagement
        if engagement_level == 'low':
            recommendations['motivation_tips'] = [
                "Set small, achievable goals",
                "Take regular breaks",
                "Find study partners",
                "Celebrate small victories"
            ]
        elif engagement_level == 'medium':
            recommendations['motivation_tips'] = [
                "Maintain consistent study schedule",
                "Mix different types of problems",
                "Track your progress",
                "Challenge yourself occasionally"
            ]
        elif engagement_level == 'high':
            recommendations['motivation_tips'] = [
                "Keep up the great work!",
                "Try more complex problems",
                "Help others learn",
                "Explore related topics"
            ]
        else:
            # Default case
            recommendations['motivation_tips'] = [
                "Stay motivated",
                "Keep practicing",
                "Set clear goals",
                "Track your progress"
            ]
        
        # Resources based on learner type and engagement
        if learner_type == 'struggling':
            recommendations['resources'] = [
                "Basic concept videos",
                "Step-by-step tutorials",
                "Practice worksheets",
                "Online calculators"
            ]
        elif learner_type == 'average':
            recommendations['resources'] = [
                "Mixed difficulty problems",
                "Concept explanations",
                "Practice tests",
                "Study guides"
            ]
        else:  # advanced
            recommendations['resources'] = [
                "Advanced problem sets",
                "Complex concept videos",
                "Competition problems",
                "Research papers"
            ]
        
        # Next steps
        if engagement_level == 'low':
            recommendations['next_steps'] = [
                "Start with 15-minute study sessions",
                "Focus on one concept at a time",
                "Use visual aids and examples",
                "Take breaks between sessions"
            ]
        elif engagement_level == 'medium':
            recommendations['next_steps'] = [
                "Increase study time gradually",
                "Mix different subjects",
                "Set weekly goals",
                "Review progress regularly"
            ]
        else:  # high
            recommendations['next_steps'] = [
                "Maintain current study pace",
                "Explore advanced topics",
                "Help others learn",
                "Set challenging goals"
            ]
        
        return recommendations

# Global model manager instance
_model_manager = None

def get_model_manager() -> AIModelManager:
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = AIModelManager()
    return _model_manager
