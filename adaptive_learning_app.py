"""
Adaptive Learning System
Clean, simple interface with magenta theme using ASSISTments dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Adaptive Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for magenta theme - NO WHITE BACKGROUNDS
st.markdown("""
<style>
    /* Main theme colors - NO WHITE BACKGROUNDS */
    .main-header {
        background: linear-gradient(135deg, #E91E63, #9C27B0) !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        color: white !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    
    .card {
        background: linear-gradient(135deg, #F8BBD9, #E1BEE7) !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border: 2px solid #E91E63 !important;
        margin: 1rem 0 !important;
        color: #333 !important;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4CAF50, #81C784) !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FF9800, #FFB74D) !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
    }
    
    .quiz-question {
        background: linear-gradient(135deg, #F8BBD9, #E1BEE7) !important;
        padding: 2rem !important;
        border-radius: 10px !important;
        border: 2px solid #E91E63 !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        color: #333 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #E91E63, #9C27B0) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #C2185B, #7B1FA2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Ensure all text is visible */
    .card h1, .card h2, .card h3, .card p {
        color: #333 !important;
    }
    
    .quiz-question h3, .quiz-question p {
        color: #333 !important;
    }
    
    .success-card h1, .success-card h2, .success-card h3, .success-card p {
        color: white !important;
    }
    
    .warning-card h1, .warning-card h2, .warning-card h3, .warning-card p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load processed data
@st.cache_data
def load_data():
    """Load processed ASSISTments data"""
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
        st.error("❌ Processed data not found. Please run the data processor first.")
        return None

# Initialize session state
if 'user_registered' not in st.session_state:
    st.session_state.user_registered = False
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Adaptive Learning System</h1>
        <p>Take a quiz and get personalized learning recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar navigation
    st.sidebar.markdown("## 📚 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📝 Take Quiz", "📊 Results", "👥 Learner Profiles", "📈 Analytics"]
    )
    
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📝 Take Quiz":
        show_quiz_page(data)
    elif page == "📊 Results":
        show_results_page(data)
    elif page == "👥 Learner Profiles":
        show_profiles_page(data)
    elif page == "📈 Analytics":
        show_analytics_page(data)

def show_home_page(data):
    """Show home page with registration"""
    
    if not st.session_state.user_registered:
        st.markdown("""
        <div class="card">
            <h2>👋 Welcome to Adaptive Learning</h2>
            <p>This system uses real educational data from ASSISTments to provide personalized learning recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("registration_form"):
            st.markdown("### 📝 Register to Start")
            
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name")
                email = st.text_input("Email")
            
            with col2:
                grade = st.selectbox("Grade Level", ["7th", "8th", "9th", "10th", "11th", "12th"])
                subject = st.selectbox("Subject", ["Mathematics"])
            
            if st.form_submit_button("🚀 Start Learning", type="primary"):
                if name and email:
                    st.session_state.user_registered = True
                    st.session_state.user_info = {
                        'name': name,
                        'email': email,
                        'grade': grade,
                        'subject': subject
                    }
                    st.success("✅ Registration successful! You can now take the quiz.")
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")
    else:
        st.markdown("""
        <div class="success-card">
            <h2>✅ Welcome back, {}!</h2>
            <p>You're registered and ready to take the adaptive quiz.</p>
        </div>
        """.format(st.session_state.user_info['name']), unsafe_allow_html=True)
        
        # col1, col2, col3 = st.columns([1, 2, 1])
        # with col2:
        #     if st.button("📝 Take Quiz Now", type="primary", use_container_width=True):
        #         st.rerun()

def show_quiz_page(data):
    """Show quiz interface"""
    
    if not st.session_state.user_registered:
        st.warning("Please register first on the Home page.")
        return
    
    st.markdown("""
    <div class="card">
        <h2>📝 Mathematics Quiz</h2>
        <p>Answer 10 questions to get your personalized learning profile.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate quiz if not exists
    if st.session_state.current_quiz is None:
        st.session_state.current_quiz = generate_quiz(data['question_bank'])
        st.session_state.quiz_results = []
        st.session_state.current_question = 0
    
    # Show current question
    if st.session_state.current_question < len(st.session_state.current_quiz):
        show_question(st.session_state.current_question)
    else:
        # Quiz completed
        analyze_results(data)
        st.success("🎉 Quiz completed! Check your results.")

def generate_quiz(question_bank, num_questions=10):
    """Generate a quiz with real math questions"""
    
    # Real math questions with answers
    math_questions = [
        {
            'question': "What is 15 + 27?",
            'options': ['40', '42', '43', '41'],
            'correct': 1,
            'difficulty': 'easy',
            'explanation': '15 + 27 = 42'
        },
        {
            'question': "What is 8 × 7?",
            'options': ['54', '56', '58', '52'],
            'correct': 1,
            'difficulty': 'easy',
            'explanation': '8 × 7 = 56'
        },
        {
            'question': "What is 3/4 + 1/2?",
            'options': ['5/4', '4/6', '1 1/4', '1 1/2'],
            'correct': 0,
            'difficulty': 'intermediate',
            'explanation': '3/4 + 1/2 = 3/4 + 2/4 = 5/4'
        },
        {
            'question': "What is 25% of 80?",
            'options': ['15', '20', '25', '30'],
            'correct': 1,
            'difficulty': 'intermediate',
            'explanation': '25% of 80 = 0.25 × 80 = 20'
        },
        {
            'question': "Solve: 2x + 5 = 13",
            'options': ['x = 3', 'x = 4', 'x = 5', 'x = 6'],
            'correct': 1,
            'difficulty': 'intermediate',
            'explanation': '2x + 5 = 13 → 2x = 8 → x = 4'
        },
        {
            'question': "What is the area of a rectangle with length 6 and width 4?",
            'options': ['20', '24', '28', '32'],
            'correct': 1,
            'difficulty': 'easy',
            'explanation': 'Area = length × width = 6 × 4 = 24'
        },
        {
            'question': "What is √16?",
            'options': ['2', '4', '8', '16'],
            'correct': 1,
            'difficulty': 'easy',
            'explanation': '√16 = 4 because 4² = 16'
        },
        {
            'question': "What is 3² × 2³?",
            'options': ['36', '72', '108', '144'],
            'correct': 1,
            'difficulty': 'intermediate',
            'explanation': '3² × 2³ = 9 × 8 = 72'
        },
        {
            'question': "Solve: 3x - 7 = 2x + 3",
            'options': ['x = 4', 'x = 5', 'x = 10', 'x = 11'],
            'correct': 2,
            'difficulty': 'hard',
            'explanation': '3x - 7 = 2x + 3 → 3x - 2x = 3 + 7 → x = 10'
        },
        {
            'question': "What is the slope of the line passing through (2,3) and (4,7)?",
            'options': ['1', '2', '3', '4'],
            'correct': 1,
            'difficulty': 'hard',
            'explanation': 'Slope = (7-3)/(4-2) = 4/2 = 2'
        },
        {
            'question': "What is 1/3 + 1/6?",
            'options': ['1/2', '2/9', '1/9', '3/6'],
            'correct': 0,
            'difficulty': 'intermediate',
            'explanation': '1/3 + 1/6 = 2/6 + 1/6 = 3/6 = 1/2'
        },
        {
            'question': "What is 20% of 150?",
            'options': ['25', '30', '35', '40'],
            'correct': 1,
            'difficulty': 'easy',
            'explanation': '20% of 150 = 0.2 × 150 = 30'
        },
        {
            'question': "What is the perimeter of a square with side length 5?",
            'options': ['15', '20', '25', '30'],
            'correct': 1,
            'difficulty': 'easy',
            'explanation': 'Perimeter = 4 × side = 4 × 5 = 20'
        },
        {
            'question': "Solve: x² - 4 = 0",
            'options': ['x = ±2', 'x = ±4', 'x = 2 only', 'x = 4 only'],
            'correct': 0,
            'difficulty': 'hard',
            'explanation': 'x² - 4 = 0 → x² = 4 → x = ±2'
        },
        {
            'question': "What is 5! (5 factorial)?",
            'options': ['20', '60', '120', '240'],
            'correct': 2,
            'difficulty': 'intermediate',
            'explanation': '5! = 5 × 4 × 3 × 2 × 1 = 120'
        }
    ]
    
    # Select random questions
    selected_questions = np.random.choice(math_questions, size=min(num_questions, len(math_questions)), replace=False)
    
    # Create quiz format
    quiz = []
    for i, q in enumerate(selected_questions):
        quiz.append({
            'problem_id': f"Q{i+1}",
            'difficulty': q['difficulty'],
            'success_rate': 0.7,  # Placeholder success rate
            'question_text': q['question'],
            'options': q['options'],
            'correct_answer': q['correct'],
            'explanation': q['explanation']
        })
    
    return quiz

def show_question(question_index):
    """Show a single quiz question"""
    question = st.session_state.current_quiz[question_index]
    
    st.markdown(f"""
    <div class="quiz-question">
        <h3>Question {question_index + 1} of 10</h3>
        <p><strong>{question['question_text']}</strong></p>
        <p>Success Rate: {question['success_rate']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    progress = (question_index + 1) / 10
    st.progress(progress)
    
    # Answer options
    st.markdown("### Select your answer:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"A) {question['options'][0]}", key=f"q{question_index}_a", use_container_width=True):
            record_answer(question_index, 0)
    
    with col2:
        if st.button(f"B) {question['options'][1]}", key=f"q{question_index}_b", use_container_width=True):
            record_answer(question_index, 1)
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button(f"C) {question['options'][2]}", key=f"q{question_index}_c", use_container_width=True):
            record_answer(question_index, 2)
    
    with col4:
        if st.button(f"D) {question['options'][3]}", key=f"q{question_index}_d", use_container_width=True):
            record_answer(question_index, 3)

def record_answer(question_index, selected_answer):
    """Record user's answer and move to next question"""
    question = st.session_state.current_quiz[question_index]
    
    # Record result
    result = {
        'question_index': question_index,
        'problem_id': question['problem_id'],
        'difficulty': question['difficulty'],
        'selected_answer': selected_answer,
        'correct_answer': question['correct_answer'],
        'is_correct': selected_answer == question['correct_answer'],
        'success_rate': question['success_rate'],
        'timestamp': datetime.now()
    }
    
    st.session_state.quiz_results.append(result)
    
    # Move to next question
    st.session_state.current_question += 1
    
    if st.session_state.current_question < len(st.session_state.current_quiz):
        st.rerun()
    else:
        st.rerun()

def analyze_results(data):
    """Analyze quiz results and create user profile"""
    if not st.session_state.quiz_results:
        return
    
    results = st.session_state.quiz_results
    
    # Calculate performance metrics
    accuracy = sum(1 for r in results if r['is_correct']) / len(results)
    
    # Classify learner type
    if accuracy >= 0.8:
        learner_type = "Advanced"
        color = "#4CAF50"
    elif accuracy >= 0.6:
        learner_type = "Moderate"
        color = "#FF9800"
    else:
        learner_type = "Struggling"
        color = "#F44336"
    
    # Create user profile
    st.session_state.user_profile = {
        'accuracy': accuracy,
        'learner_type': learner_type,
        'total_questions': len(results),
        'correct_answers': sum(1 for r in results if r['is_correct']),
        'difficulty_distribution': {
            'easy': len([r for r in results if r['difficulty'] == 'easy']),
            'intermediate': len([r for r in results if r['difficulty'] == 'intermediate']),
            'hard': len([r for r in results if r['difficulty'] == 'hard'])
        }
    }
    
    # Show results
    st.markdown(f"""
    <div class="success-card">
        <h2>🎯 Your Learning Profile</h2>
        <h3 style="color: {color};">{learner_type} Learner</h3>
        <p>Accuracy: {accuracy:.1%} ({st.session_state.user_profile['correct_answers']}/{len(results)} correct)</p>
    </div>
    """, unsafe_allow_html=True)

def show_results_page(data):
    """Show detailed results and recommendations"""
    if not st.session_state.user_profile:
        st.warning("Please complete the quiz first.")
        return
    
    profile = st.session_state.user_profile
    
    st.markdown("## 📊 Your Results")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{profile['accuracy']:.1%}")
    with col2:
        st.metric("Learner Type", profile['learner_type'])
    with col3:
        st.metric("Questions Answered", profile['total_questions'])
    
    # Performance chart
    if st.session_state.quiz_results:
        results_df = pd.DataFrame(st.session_state.quiz_results)
        
        fig = px.line(
            results_df, 
            x='question_index', 
            y='is_correct',
            title="Performance Over Time",
            labels={'question_index': 'Question Number', 'is_correct': 'Correct (1) / Incorrect (0)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("## 💡 Personalized Recommendations")
    
    if profile['learner_type'] == "Advanced":
        st.markdown("""
        <div class="success-card">
            <h3>🚀 Advanced Learner Recommendations</h3>
            <ul>
                <li>Focus on challenging problems and advanced topics</li>
                <li>Explore problem-solving strategies and multiple approaches</li>
                <li>Consider mentoring other students</li>
                <li>Take on complex, multi-step problems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif profile['learner_type'] == "Moderate":
        st.markdown("""
        <div class="warning-card">
            <h3>📈 Moderate Learner Recommendations</h3>
            <ul>
                <li>Practice with intermediate difficulty problems</li>
                <li>Focus on building confidence with fundamentals</li>
                <li>Gradually increase difficulty level</li>
                <li>Review concepts you find challenging</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h3>🎯 Struggling Learner Recommendations</h3>
            <ul>
                <li>Start with basic concepts and fundamentals</li>
                <li>Take your time and don't rush</li>
                <li>Ask for help when needed</li>
                <li>Practice with easier problems to build confidence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_profiles_page(data):
    """Show learner profiles from the dataset"""
    st.markdown("## 👥 Learner Profiles from ASSISTments Dataset")
    
    profiles = data['learner_profiles']
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Learners", len(profiles))
    with col2:
        st.metric("Average Accuracy", f"{profiles['accuracy'].mean():.1%}")
    with col3:
        st.metric("Advanced Learners", len(profiles[profiles['learner_type'] == 'advanced']))
    with col4:
        st.metric("Struggling Learners", len(profiles[profiles['learner_type'] == 'struggling']))
    
    # Learner type distribution
    fig = px.pie(
        profiles, 
        names='learner_type', 
        title="Distribution of Learner Types"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show sample profiles
    st.markdown("### 📋 Sample Learner Profiles")
    st.dataframe(profiles.head(10))

def show_analytics_page(data):
    """Show analytics and insights"""
    st.markdown("## 📈 Dataset Analytics")
    
    clean_data = data['clean_data']
    question_bank = data['question_bank']
    
    # Question difficulty distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            question_bank, 
            names='difficulty', 
            title="Question Difficulty Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            question_bank, 
            x='success_rate',
            title="Success Rate Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    st.markdown("### 📊 Performance Insights")
    
    # Average performance by difficulty
    difficulty_stats = question_bank.groupby('difficulty').agg({
        'success_rate': 'mean',
        'total_attempts': 'mean'
    }).round(3)
    
    st.dataframe(difficulty_stats)

if __name__ == "__main__":
    main() 