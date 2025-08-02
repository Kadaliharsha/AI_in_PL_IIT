"""
ASSISTments Dataset Processor
Handles loading, cleaning, and preprocessing of ASSISTments educational data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ASSISTmentsProcessor:
    """Process ASSISTments dataset for adaptive learning system"""
    
    def __init__(self, data_path: str = "data/raw/"):
        self.data_path = data_path
        self.raw_data = None
        self.problems_data = None
        self.students_data = None
        self.processed_data = None
        
    def load_raw_data(self) -> bool:
        """Load raw ASSISTments data files"""
        try:
            # Try to load main data file
            main_file = os.path.join(self.data_path, "skill_builder_data.csv")
            if os.path.exists(main_file):
                print(f"Loading main data from: {main_file}")
                self.raw_data = pd.read_csv(main_file)
            else:
                # Try alternative filenames
                possible_files = [
                    "assistments_data.csv",
                    "skill_builder_2009_2010.csv",
                    "assistments_2009_2010.csv",
                    "data.csv"
                ]
                
                for file in possible_files:
                    file_path = os.path.join(self.data_path, file)
                    if os.path.exists(file_path):
                        print(f"Loading data from: {file_path}")
                        self.raw_data = pd.read_csv(file_path)
                        break
                else:
                    print("❌ No ASSISTments data file found!")
                    print("Please place your downloaded CSV file in the data/raw/ directory")
                    return False
            
            # Load problems metadata if available
            problems_file = os.path.join(self.data_path, "problems.csv")
            if os.path.exists(problems_file):
                print("Loading problems metadata...")
                self.problems_data = pd.read_csv(problems_file)
            
            # Load students metadata if available
            students_file = os.path.join(self.data_path, "students.csv")
            if os.path.exists(students_file):
                print("Loading students metadata...")
                self.students_data = pd.read_csv(students_file)
            
            print(f"✅ Successfully loaded data with {len(self.raw_data)} records")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def explore_data(self) -> Dict:
        """Explore the loaded data structure"""
        if self.raw_data is None:
            print("❌ No data loaded. Call load_raw_data() first.")
            return {}
        
        exploration = {
            "total_records": len(self.raw_data),
            "columns": list(self.raw_data.columns),
            "data_types": self.raw_data.dtypes.to_dict(),
            "missing_values": self.raw_data.isnull().sum().to_dict(),
            "unique_students": self.raw_data['user_id'].nunique() if 'user_id' in self.raw_data.columns else 0,
            "unique_problems": self.raw_data['problem_id'].nunique() if 'problem_id' in self.raw_data.columns else 0,
            "sample_data": self.raw_data.head(5).to_dict('records')
        }
        
        print("📊 Data Exploration Results:")
        print(f"Total records: {exploration['total_records']}")
        print(f"Unique students: {exploration['unique_students']}")
        print(f"Unique problems: {exploration['unique_problems']}")
        print(f"Columns: {exploration['columns']}")
        
        return exploration
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the raw data"""
        if self.raw_data is None:
            print("❌ No data loaded. Call load_raw_data() first.")
            return pd.DataFrame()
        
        print("🧹 Cleaning data...")
        
        # Make a copy to avoid modifying original
        clean_data = self.raw_data.copy()
        
        # Standardize column names for ASSISTments 2009-2010 dataset
        column_mapping = {
            'user_id': 'student_id',
            'problem_id': 'problem_id',  # Use problem_id directly
            'correct': 'correct',
            'attempt_count': 'attempts',
            'ms_first_response_time': 'time_taken_ms',
            'list_skills': 'skill_id',
            'order_id': 'order_id'
        }
        
        # Rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in clean_data.columns}
        clean_data = clean_data.rename(columns=existing_columns)
        
        # Handle missing values
        if 'correct' in clean_data.columns:
            clean_data['correct'] = clean_data['correct'].fillna(0).astype(int)
        
        if 'attempts' in clean_data.columns:
            clean_data['attempts'] = clean_data['attempts'].fillna(1).astype(int)
        
        if 'time_taken_ms' in clean_data.columns:
            # Convert milliseconds to seconds and handle outliers
            clean_data['time_taken_seconds'] = clean_data['time_taken_ms'] / 1000
            # Remove extreme outliers (more than 30 minutes)
            clean_data = clean_data[clean_data['time_taken_seconds'] <= 1800]
        
        # Add hints_used column (not available in this dataset, set to 0)
        clean_data['hints_used'] = 0
        
        # Ensure problem_id is clean (remove any problematic values)
        clean_data = clean_data.dropna(subset=['problem_id'])
        clean_data['problem_id'] = clean_data['problem_id'].astype(str)
        
        # Add derived features
        clean_data['date'] = pd.to_datetime('now')  # Placeholder for actual dates
        
        # Calculate performance metrics
        clean_data['accuracy'] = clean_data['correct']
        clean_data['efficiency'] = clean_data['correct'] / clean_data['attempts']
        
        print(f"✅ Cleaned data: {len(clean_data)} records")
        return clean_data
    
    def extract_learner_features(self, student_data: pd.DataFrame) -> Dict:
        """Extract features for a specific student"""
        if len(student_data) == 0:
            return {}
        
        features = {
            'student_id': student_data['student_id'].iloc[0],
            'total_questions': len(student_data),
            'accuracy': student_data['correct'].mean(),
            'avg_time_seconds': student_data.get('time_taken_seconds', pd.Series([60])).mean(),
            'avg_attempts': student_data['attempts'].mean(),
            'avg_hints_used': student_data.get('hints_used', pd.Series([0])).mean(),
            'consistency': 1 - student_data['correct'].std(),  # Lower std = higher consistency
            'engagement': len(student_data) / 10,  # Normalize by expected questions
            'efficiency': student_data.get('efficiency', student_data['correct']).mean()
        }
        
        return features
    
    def create_learner_profiles(self, clean_data: pd.DataFrame, min_questions: int = 5) -> pd.DataFrame:
        """Create learner profiles from cleaned data"""
        print("👥 Creating learner profiles...")
        
        # Group by student and filter those with enough data
        student_groups = clean_data.groupby('student_id').filter(lambda x: len(x) >= min_questions)
        
        if len(student_groups) == 0:
            print("❌ No students with enough data found")
            return pd.DataFrame()
        
        # Extract features for each student
        learner_profiles = []
        for student_id, student_data in student_groups.groupby('student_id'):
            features = self.extract_learner_features(student_data)
            if features:
                learner_profiles.append(features)
        
        profiles_df = pd.DataFrame(learner_profiles)
        
        # Classify learner types
        profiles_df['learner_type'] = self.classify_learner_types(profiles_df)
        
        print(f"✅ Created {len(profiles_df)} learner profiles")
        return profiles_df
    
    def classify_learner_types(self, profiles_df: pd.DataFrame) -> List[str]:
        """Classify learners into types based on performance patterns"""
        learner_types = []
        
        for _, row in profiles_df.iterrows():
            accuracy = row['accuracy']
            avg_time = row['avg_time_seconds']
            hints = row['avg_hints_used']
            
            if accuracy >= 0.8 and avg_time <= 60 and hints <= 0.5:
                learner_type = 'advanced'
            elif accuracy >= 0.6 and avg_time <= 120:
                learner_type = 'moderate'
            elif accuracy < 0.4 or avg_time > 180 or hints > 2:
                learner_type = 'struggling'
            else:
                learner_type = 'balanced'
            
            learner_types.append(learner_type)
        
        return learner_types
    
    def create_question_bank(self, clean_data: pd.DataFrame) -> pd.DataFrame:
        """Create a question bank from the data"""
        print("📚 Creating question bank...")
        
        if 'problem_id' not in clean_data.columns:
            print("❌ No problem_id column found")
            return pd.DataFrame()
        
        question_stats = clean_data.groupby('problem_id').agg({
            'correct': ['count', 'mean'],
            'attempts': 'mean'
        }).round(3)
        
        question_stats.columns = ['total_attempts', 'success_rate', 'avg_attempts']
        question_stats = question_stats.reset_index()
        
        # Add difficulty classification
        question_stats['difficulty'] = question_stats['success_rate'].apply(
            lambda x: 'easy' if x >= 0.7 else 'intermediate' if x >= 0.4 else 'hard'
        )
        
        print(f"✅ Created question bank with {len(question_stats)} questions")
        return question_stats
    
    def save_processed_data(self, clean_data: pd.DataFrame, profiles_df: pd.DataFrame, 
                          question_bank: pd.DataFrame) -> bool:
        """Save processed data to files"""
        try:
            # Create processed directory if it doesn't exist
            os.makedirs("data/processed", exist_ok=True)
            
            # Save cleaned data
            clean_data.to_csv("data/processed/clean_assistments_data.csv", index=False)
            print("✅ Saved clean data")
            
            # Save learner profiles
            profiles_df.to_csv("data/processed/learner_profiles.csv", index=False)
            print("✅ Saved learner profiles")
            
            # Save question bank
            question_bank.to_csv("data/processed/question_bank.csv", index=False)
            print("✅ Saved question bank")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False
    
    def process_full_pipeline(self) -> bool:
        """Run the complete data processing pipeline"""
        print("🚀 Starting ASSISTments data processing pipeline...")
        
        # Step 1: Load raw data
        if not self.load_raw_data():
            return False
        
        # Step 2: Explore data
        exploration = self.explore_data()
        if not exploration:
            return False
        
        # Step 3: Clean data
        clean_data = self.clean_data()
        if len(clean_data) == 0:
            return False
        
        # Step 4: Create learner profiles
        profiles_df = self.create_learner_profiles(clean_data)
        if len(profiles_df) == 0:
            return False
        
        # Step 5: Create question bank
        question_bank = self.create_question_bank(clean_data)
        
        # Step 6: Save processed data
        if not self.save_processed_data(clean_data, profiles_df, question_bank):
            return False
        
        print("🎉 Data processing pipeline completed successfully!")
        return True

# Utility function to get processed data
def load_processed_assistments_data() -> Dict:
    """Load processed ASSISTments data for the adaptive learning system"""
    try:
        clean_data = pd.read_csv("data/processed/clean_assistments_data.csv")
        profiles = pd.read_csv("data/processed/learner_profiles.csv")
        question_bank = pd.read_csv("data/processed/question_bank.csv")
        
        return {
            'clean_data': clean_data,
            'learner_profiles': profiles,
            'question_bank': question_bank
        }
    except FileNotFoundError:
        print("❌ Processed data not found. Run the processor first.")
        return {}

if __name__ == "__main__":
    # Run the processor
    processor = ASSISTmentsProcessor()
    success = processor.process_full_pipeline()
    
    if success:
        print("✅ ASSISTments data ready for adaptive learning system!")
    else:
        print("❌ Failed to process ASSISTments data") 