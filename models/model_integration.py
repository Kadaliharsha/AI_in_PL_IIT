"""
Model Integration Module for Real-time AI Predictions

This module loads the trained models and provides functions for:
1. Real-time learner classification
2. Performance prediction for next question
3. Engagement analysis
4. Adaptive question selection
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class AIModelManager:
    """Manages all trained AI models for real-time predictions"""
    
    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.artifacts_dir = artifacts_dir
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self) -> None:
        """Load all trained models from artifacts directory"""
        print("🤖 Loading trained AI models...")
        
        # Only load the new models we created with simple_trainer.py
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
                    
                    # Extract only the pipeline and essential data
                    if isinstance(model_data, dict) and 'pipeline' in model_data:
                        self.models[model_name] = {
                            'pipeline': model_data['pipeline'],
                            'feature_names': model_data.get('feature_names', []),
                            'classes': model_data.get('classes', []),
                            'accuracy': model_data.get('accuracy', 0.0)
                        }
                        print(f"✅ Loaded {model_name}")
                    else:
                        print(f"⚠️ Invalid model format for {model_file}")
                        
                except Exception as e:
                    print(f"❌ Error loading {model_file}: {e}")
                    # Try alternative loading method
                    try:
                        with open(model_path, 'rb') as f:
                            # Try to load with different pickle protocol
                            model_data = pickle.load(f, encoding='latin1')
                            if isinstance(model_data, dict) and 'pipeline' in model_data:
                                self.models[model_name] = {
                                    'pipeline': model_data['pipeline'],
                                    'feature_names': model_data.get('feature_names', []),
                                    'classes': model_data.get('classes', []),
                                    'accuracy': model_data.get('accuracy', 0.0)
                                }
                                print(f"✅ Loaded {model_name} (alternative method)")
                    except:
                        print(f"❌ Failed to load {model_file} with alternative method")
            else:
                print(f"⚠️ Model file not found: {model_file}")
        
        print(f"🎯 Loaded {len(self.models)} models successfully!")
    
    def predict_learner_type(self, student_features: Dict) -> Dict:
        """Predict student's learning type using trained models"""
        
        if 'learner_classification_rf' not in self.models:
            return {"error": "Learner classification model not loaded"}
        
        try:
            # Extract features in the same order as training
            feature_cols = [
                'accuracy', 'total_questions', 'avg_time_seconds', 
                'avg_attempts', 'avg_hints_used', 'consistency', 
                'speed_accuracy_tradeoff', 'persistence', 'engagement', 'efficiency'
            ]
            
            # Create feature vector with proper feature names
            X = pd.DataFrame([[student_features.get(col, 0) for col in feature_cols]], columns=feature_cols)
            
            # Get predictions from RF model
            rf_model = self.models['learner_classification_rf']['pipeline']
            rf_pred = rf_model.predict(X)[0]
            rf_proba = rf_model.predict_proba(X)[0]
            
            # Use RF prediction
            final_prediction = rf_pred
            confidence = max(rf_proba)
            
            return {
                "learner_type": final_prediction,
                "confidence": confidence,
                "rf_prediction": rf_pred,
                "rf_confidence": max(rf_proba)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_performance(self, question_features: Dict) -> Dict:
        """Predict success probability for next question"""
        
        if 'performance_prediction_gb' not in self.models:
            return {"error": "Performance prediction model not loaded"}
        
        try:
            # Extract features in the same order as training (only available features)
            feature_cols = [
                'attempts', 'time_taken_seconds', 'hints_used', 
                'attempt_efficiency'
            ]
            
            # Create feature vector with proper feature names
            X = pd.DataFrame([[question_features.get(col, 0) for col in feature_cols]], columns=feature_cols)
            
            # Get predictions from GB model
            gb_model = self.models['performance_prediction_gb']['pipeline']
            gb_proba = gb_model.predict_proba(X)[0]
            
            success_probability = gb_proba[1]
            
            return {
                "success_probability": success_probability,
                "gb_probability": gb_proba[1],
                "predicted_success": success_probability > 0.5
            }
            
        except Exception as e:
            return {"error": f"Performance prediction failed: {str(e)}"}
    
    def analyze_engagement(self, behavior_features: Dict) -> Dict:
        """Analyze student engagement level"""
        
        if 'engagement_analysis_rf' not in self.models:
            return {"error": "Engagement analysis model not loaded"}
        
        try:
            # Extract features in the same order as training
            feature_cols = [
                'total_interactions', 'avg_accuracy', 'accuracy_std', 
                'avg_attempts', 'avg_time'
            ]
            
            # Create feature vector with proper feature names
            X = pd.DataFrame([[behavior_features.get(col, 0) for col in feature_cols]], columns=feature_cols)
            
            # Get predictions from RF model
            rf_model = self.models['engagement_analysis_rf']['pipeline']
            rf_pred = rf_model.predict(X)[0]
            rf_proba = rf_model.predict_proba(X)[0]
            
            # Use RF prediction
            final_prediction = rf_pred
            confidence = max(rf_proba)
            
            return {
                "engagement_level": final_prediction,
                "confidence": confidence,
                "rf_prediction": rf_pred
            }
            
        except Exception as e:
            return {"error": f"Engagement analysis failed: {str(e)}"}
    
    def get_adaptive_recommendations(self, student_data: Dict) -> Dict:
        """Generate adaptive learning recommendations based on AI analysis"""
        
        # Get all predictions
        learner_result = self.predict_learner_type(student_data)
        engagement_result = self.analyze_engagement(student_data)
        
        if "error" in learner_result or "error" in engagement_result:
            return {"error": "Failed to generate recommendations"}
        
        learner_type = learner_result["learner_type"]
        engagement_level = engagement_result["engagement_level"]
        
        # Generate personalized recommendations
        recommendations = self._generate_recommendations(learner_type, engagement_level, student_data)
        
        return {
            "learner_type": learner_type,
            "engagement_level": engagement_level,
            "recommendations": recommendations,
            "confidence": {
                "learner": learner_result["confidence"],
                "engagement": engagement_result["confidence"]
            }
        }
    
    def _generate_recommendations(self, learner_type: str, engagement_level: str, student_data: Dict) -> Dict:
        """Generate personalized learning recommendations"""
        
        recommendations = {
            "study_plan": [],
            "resources": [],
            "difficulty_adjustment": "",
            "motivation_tips": []
        }
        
        # Study plan based on learner type
        if learner_type == "advanced":
            recommendations["study_plan"] = [
                "Focus on challenging problems and advanced topics",
                "Explore multiple solution approaches",
                "Consider mentoring other students",
                "Take on complex, multi-step problems"
            ]
            recommendations["difficulty_adjustment"] = "Increase difficulty to maintain engagement"
            
        elif learner_type == "moderate":
            recommendations["study_plan"] = [
                "Practice with intermediate difficulty problems",
                "Focus on building confidence with fundamentals",
                "Gradually increase difficulty level",
                "Review concepts you find challenging"
            ]
            recommendations["difficulty_adjustment"] = "Maintain current difficulty with gradual increases"
            
        elif learner_type == "struggling":
            recommendations["study_plan"] = [
                "Start with basic concepts and fundamentals",
                "Take your time and don't rush",
                "Ask for help when needed",
                "Practice with easier problems to build confidence"
            ]
            recommendations["difficulty_adjustment"] = "Decrease difficulty and provide more support"
        
        # Engagement-based recommendations
        if engagement_level == "low":
            recommendations["motivation_tips"] = [
                "Set small, achievable goals",
                "Take regular breaks",
                "Find study partners or groups",
                "Reward yourself for progress"
            ]
        elif engagement_level == "high":
            recommendations["motivation_tips"] = [
                "Maintain your excellent momentum!",
                "Challenge yourself with advanced topics",
                "Share your knowledge with others",
                "Set ambitious but realistic goals"
            ]
        
        # Resource recommendations
        recommendations["resources"] = [
            "Interactive practice problems",
            "Video tutorials",
            "Peer study groups",
            "One-on-one tutoring sessions"
        ]
        
        return recommendations
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {}
        for model_name, model_data in self.models.items():
            info[model_name] = {
                "accuracy": model_data.get("accuracy", "N/A"),
                "features": len(model_data.get("feature_names", [])),
                "classes": model_data.get("classes", "N/A"),
            }
        return info

# Global model manager instance
model_manager = None

def get_model_manager() -> AIModelManager:
    """Get or create the global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = AIModelManager()
    return model_manager
