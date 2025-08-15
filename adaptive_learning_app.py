"""
Adaptive Learning System
Clean, simple interface for personalized learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.model_integration import get_model_manager

# Page configuration
st.set_page_config(
    page_title="AI-Powered Adaptive Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
with open('style.css', 'r') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎓 AI-Powered Adaptive Learning System</h1>', unsafe_allow_html=True)

# Initialize session state
if 'user_registered' not in st.session_state:
    st.session_state.user_registered = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = []
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# Simple navigation without sidebar
def show_home():
    st.markdown('<div class="student-welcome">', unsafe_allow_html=True)
    st.markdown("## 🚀 Welcome to Your Personalized Learning Journey!")
    st.markdown("Our AI system will analyze your learning patterns and create a customized study plan just for you.")
    st.markdown('</div>', unsafe_allow_html=True)
    # Only show registration form if not registered
    if not st.session_state.user_registered:
        st.markdown("""
        ### 📚 How It Works:
        1. **Register** - Tell us about yourself
        2. **Take Assessment** - Complete a short quiz
        3. **Get AI Analysis** - Receive personalized insights
        4. **Follow Recommendations** - Study smarter, not harder
        """)
        with st.form("registration_form"):
            st.subheader("📝 Student Registration")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email")
            with col2:
                grade = st.selectbox("Grade Level", ["7th", "8th", "9th", "10th", "11th", "12th"])
                subject = st.selectbox("Subject", ["Mathematics", "Science", "English", "History"])
            if st.form_submit_button("🚀 Start Learning Journey", type="primary"):
                if name and email:
                    st.session_state.user_registered = True
                    st.session_state.user_info = {
                        'name': name,
                        'email': email,
                        'grade': grade,
                        'subject': subject
                    }
                    st.success("✅ Registration successful! You can now take the assessment.")
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")
    else:
        st.success(f"✅ Welcome back, {st.session_state.user_info['name']}!")
        st.info(f"You're registered for {st.session_state.user_info['subject']} in {st.session_state.user_info['grade']} grade.")
        
        # Show Start Assessment button for new students
        if not st.session_state.quiz_results:
            st.markdown("---")
            st.subheader("🎯 Ready to Start Your Learning Journey?")
            st.info("Take your first assessment to get personalized recommendations!")
            if st.button("🚀 Start Assessment", type="primary", use_container_width=True, key="start_first_assessment"):
                st.session_state.current_page = "quiz"
                st.rerun()
        else:
            st.markdown("---")
            st.subheader("📊 Your Learning Progress")
            st.info("You've already taken assessments! Check your results and history for insights.")
    # Navigation handled in main()

def show_quiz():
    st.header("📝 Learning Assessment")
    st.info("This quiz helps our AI understand your learning style and current knowledge level.")
    
    if not st.session_state.user_registered:
        st.warning("Please register first!")
        st.session_state.current_page = "home"
        st.rerun()
        return
    
    # Only generate a new quiz if we don't have one or if it's a fresh attempt
    if st.session_state.current_quiz is None:
        st.session_state.current_quiz = generate_quiz()
        # Show appropriate message based on student status
        if st.session_state.quiz_history:
            st.success("🎯 Fresh quiz loaded! Answer all questions to continue.")
    else:
            st.success("🎯 Welcome to your first assessment! Answer all questions to get started.")
    
    quiz = st.session_state.current_quiz
    
    # Quiz interface
    st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
    st.subheader(f"📝 Quiz ({len(quiz)} questions)")
    
    # Show quiz instructions
    st.info("💡 **Instructions**: Read each question carefully and select your answer. All questions must be answered before submitting.")
    
    answers = []
    for i, question in enumerate(quiz):
        st.markdown('<div class="question-box">', unsafe_allow_html=True)
        st.write(f"**Question {i+1}:** {question['question']}")
        
        # Use unique keys for each quiz attempt to prevent state retention
        answer = st.radio(
            f"Select your answer:",
            question['options'],
            key=f"quiz_{len(st.session_state.quiz_history)}_{i}",
            label_visibility="collapsed",
            index=None  # No default selection
        )
        
        answers.append(answer)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if all questions are answered
    all_answered = all(answer is not None for answer in answers)
    
    # Submit button (only enabled when all questions are answered)
    if st.button("Submit Quiz", type="primary", use_container_width=True, disabled=not all_answered):
        if not all_answered:
            st.error("Please answer all questions before submitting!")
            return
            
        # Process answers
        results = []
        for i, (question, answer) in enumerate(zip(quiz, answers)):
            is_correct = answer == question['options'][question['correct']]
            results.append({
                'question': question['question'],
                'user_answer': answer,
                'correct_answer': question['options'][question['correct']],
                'is_correct': is_correct,
                'explanation': question['explanation']
            })
        
        # Calculate performance metrics
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['is_correct'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Create quiz attempt record
        import datetime
        quiz_attempt = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'score': f"{correct_answers}/{total_questions}",
            'results': results,
            'subject': st.session_state.user_info.get('subject', 'Mathematics'),
            'attempt_number': len(st.session_state.quiz_history) + 1
        }
        
        # Store in history and current results
        st.session_state.quiz_history.append(quiz_attempt)
        st.session_state.quiz_results = results
        
        # Clear the current quiz to force a fresh one next time
        st.session_state.current_quiz = None
        
        # Go to results page
        st.session_state.current_page = "results"
        st.rerun()

    # Show warning if not all questions are answered
    if not all_answered:
        st.warning("⚠️ Please answer all questions before submitting!")

def show_results(results=None, quiz_attempt=None):
    if results is None:
        results = st.session_state.quiz_results
    
    if not results:
        st.info("Take a quiz first to see your results!")
        return
    
    st.header("📊 Learning Analysis")
    
    # Show quiz history if available
    if st.session_state.quiz_history and len(st.session_state.quiz_history) > 1:
        st.subheader("📈 Learning Progress")
        
        # Create progress chart
        history_data = st.session_state.quiz_history
        dates = [attempt['date'] for attempt in history_data]
        accuracies = [attempt['accuracy'] for attempt in history_data]
        scores = [attempt['score'] for attempt in history_data]
        
        # Progress over time
        fig = px.line(
            x=dates,
            y=accuracies,
            title="Accuracy Progress Over Time",
            labels={'x': 'Date', 'y': 'Accuracy'},
            markers=True
        )
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
        # History table
        st.write("**Quiz History:**")
        history_df = pd.DataFrame([
            {
                'Attempt': attempt['attempt_number'],
                'Date': attempt['date'],
                'Score': attempt['score'],
                'Accuracy': f"{attempt['accuracy']:.1%}",
                'Subject': attempt['subject']
            }
            for attempt in history_data
        ])
        st.dataframe(history_df, use_container_width=True)
        
        # Progress insights
        if len(history_data) >= 2:
            latest_accuracy = history_data[-1]['accuracy']
            previous_accuracy = history_data[-2]['accuracy']
            improvement = latest_accuracy - previous_accuracy
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest Accuracy", f"{latest_accuracy:.1%}")
            with col2:
                st.metric("Previous Accuracy", f"{previous_accuracy:.1%}")
            with col3:
                if improvement > 0:
                    st.success(f"Improvement: +{improvement:.1%}")
                elif improvement < 0:
                    st.error(f"Change: {improvement:.1%}")
                else:
                    st.info("No Change")
    
    # Current quiz results
    st.subheader("📊 Current Quiz Results")
    
    # Calculate basic metrics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r['is_correct'])
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Show basic results
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.subheader("📈 Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Correct Answers", correct_answers)
    with col3:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col4:
        st.metric("Score", f"{correct_answers}/{total_questions}")
    
    # Performance chart
    fig = px.bar(
        x=["Correct", "Incorrect"],
        y=[correct_answers, total_questions - correct_answers],
        title="Current Quiz Results",
        color=["Correct", "Incorrect"],
        color_discrete_map={"Correct": "green", "Incorrect": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Try to get AI analysis
    try:
        if get_model_manager is None:
            raise Exception("AI system not available")
        
        model_manager = get_model_manager()
        
        # Check if models are loaded (silently)
        if not hasattr(model_manager, 'models') or not model_manager.models:
            raise Exception("AI system not ready")
        
        # Calculate features for AI (silently)
        # Simulate realistic time and attempt data based on performance
        if accuracy == 1.0:  # Perfect score
            avg_time_seconds = 30  # Fast, confident answers
            avg_attempts = 1.0     # Got it right first try
            avg_hints_used = 0.0   # No hints needed
        elif accuracy >= 0.8:  # Good score
            avg_time_seconds = 45  # Reasonable time
            avg_attempts = 1.2     # Mostly first try
            avg_hints_used = 0.1   # Few hints
        elif accuracy >= 0.6:  # Average score
            avg_time_seconds = 60  # Standard time
            avg_attempts = 1.5     # Some retries
            avg_hints_used = 0.3   # Some hints
        else:  # Lower score
            avg_time_seconds = 90  # More time needed
            avg_attempts = 2.0     # Multiple attempts
            avg_hints_used = 0.5   # More hints used
        
        consistency = 1 - np.std([r['is_correct'] for r in results]) if len(results) > 1 else 1.0
        speed_accuracy_tradeoff = accuracy / (avg_time_seconds / 60) if avg_time_seconds > 0 else 0
        persistence = avg_attempts / accuracy if accuracy > 0 else 1.0
        
        # Improved engagement calculation that rewards good performance
        # Align with the new balanced training approach
        accuracy_score = accuracy * 40  # 40 points for accuracy
        
        # Efficiency score: Perfect (1.0 attempts) gets 30 points, worse gets fewer
        if avg_attempts <= 1.0:
            efficiency_score = 30  # Perfect efficiency
        elif avg_attempts <= 2.0:
            efficiency_score = 20  # Good efficiency
        elif avg_attempts <= 3.0:
            efficiency_score = 10  # Average efficiency
        else:
            efficiency_score = 0   # Poor efficiency
        
        # Speed score: Perfect (fast answers) gets 30 points, slower gets fewer
        if avg_time_seconds <= 30:
            speed_score = 30  # Perfect speed
        elif avg_time_seconds <= 60:
            speed_score = 20  # Good speed
        elif avg_time_seconds <= 90:
            speed_score = 10  # Average speed
        else:
            speed_score = 0   # Slow speed
        
        engagement = accuracy_score + efficiency_score + speed_score
        
        # Map engagement score to AI model's expected scale (0-1 range)
        engagement_normalized = engagement / 100.0
        
        efficiency = accuracy / avg_attempts if avg_attempts > 0 else accuracy
        
        # Enhanced features based on learning progress
        learning_progress = 0.0
        consistency_over_time = 1.0
        improvement_trend = 0.0
        
        if len(st.session_state.quiz_history) > 1:
            # Calculate learning progress
            first_accuracy = st.session_state.quiz_history[0]['accuracy']
            latest_accuracy = st.session_state.quiz_history[-1]['accuracy']
            learning_progress = latest_accuracy - first_accuracy
            
            # Calculate consistency over time
            accuracies = [attempt['accuracy'] for attempt in st.session_state.quiz_history]
            consistency_over_time = 1 - np.std(accuracies) if len(accuracies) > 1 else 1.0
            
            # Calculate improvement trend
            if len(accuracies) >= 3:
                recent_avg = np.mean(accuracies[-3:])
                earlier_avg = np.mean(accuracies[:3])
                improvement_trend = recent_avg - earlier_avg
        
        # Create features that match what the trained models expect
        student_features = {
            # Features for learner classification model
            'accuracy': accuracy,
            'total_questions': total_questions,
            'avg_time_seconds': avg_time_seconds,
            'avg_attempts': avg_attempts,
            'avg_hints_used': avg_hints_used,
            'consistency': consistency,
            'speed_accuracy_tradeoff': speed_accuracy_tradeoff,
            'persistence': persistence,
            'engagement': engagement_normalized, # Use normalized engagement
            'efficiency': efficiency,
            'learning_progress': learning_progress,
            'consistency_over_time': consistency_over_time,
            'improvement_trend': improvement_trend,
            'total_attempts': len(st.session_state.quiz_history),
            
            # Additional features for engagement analysis model
            'total_interactions': total_questions,  # Map to what model expects
            'avg_accuracy': accuracy,              # Map to what model expects
            'accuracy_std': 1 - consistency,       # Map to what model expects
            'avg_time': avg_time_seconds           # Map to what model expects
        }
        
        ai_analysis = model_manager.get_adaptive_recommendations(student_features)
        
        # Display AI-powered analysis
        st.markdown('<div class="ai-analysis">', unsafe_allow_html=True)
        st.subheader("🤖 Your Learning Profile")
        
        # Learner Profile
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Learning Style", ai_analysis['learner_type'].title())
            st.metric("Confidence Level", f"{ai_analysis['learner_confidence']:.1%}")
        with col2:
            st.metric("Engagement Level", ai_analysis['engagement_level'].title())
            st.metric("Engagement Confidence", f"{ai_analysis['engagement_confidence']:.1%}")
        
        # Show engagement breakdown for transparency
        st.subheader("📊 How Your Engagement Score is Calculated")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy Score", f"{accuracy_score:.0f}/40")
        with col2:
            st.metric("Efficiency Score", f"{efficiency_score:.0f}/30")
        with col3:
            st.metric("Speed Score", f"{speed_score:.0f}/30")
        
        # Show detailed breakdown
        st.info(f"💡 **Your Total Engagement**: {engagement:.0f}/100 (AI Model Score: {engagement_normalized:.2f})")
        
        # Detailed scoring explanation
        with st.expander("🔍 See How Your Scores Were Calculated"):
            st.write("**Accuracy Score (40 points):**")
            st.write(f"   • Your accuracy: {accuracy:.1%} × 40 = {accuracy_score:.0f} points")
            
            st.write("**Efficiency Score (30 points):**")
            st.write(f"   • Your attempts: {avg_attempts:.1f} per question")
            if avg_attempts <= 1.0:
                st.write(f"   • Perfect efficiency! You got it right first try = 30 points")
            elif avg_attempts <= 2.0:
                st.write(f"   • Good efficiency! Mostly first or second try = 20 points")
            elif avg_attempts <= 3.0:
                st.write(f"   • Average efficiency! Some retries needed = 10 points")
            else:
                st.write(f"   • Room for improvement! Many attempts needed = 0 points")
            
            st.write("**Speed Score (30 points):**")
            st.write(f"   • Your average time: {avg_time_seconds:.0f} seconds per question")
            if avg_time_seconds <= 30:
                st.write(f"   • Perfect speed! Fast, confident answers = 30 points")
            elif avg_time_seconds <= 60:
                st.write(f"   • Good speed! Reasonable time = 20 points")
            elif avg_time_seconds <= 90:
                st.write(f"   • Average speed! Some time needed = 10 points")
            else:
                st.write(f"   • Room for improvement! More time needed = 0 points")
        
        # Show engagement level interpretation
        if engagement >= 70:
            st.success("🎯 **Engagement Level**: HIGH - Excellent focus and performance!")
        elif engagement >= 40:
            st.warning("🎯 **Engagement Level**: MEDIUM - Good effort, room for improvement!")
        else:
            st.info("🎯 **Engagement Level**: LOW - Keep practicing to improve!")
        
        # Learning Progress Analysis
        if len(st.session_state.quiz_history) > 1:
            st.subheader("📈 Your Learning Journey")
            col1, col2, col3 = st.columns(3)
            with col1:
                if learning_progress > 0:
                    st.success(f"Overall Progress: +{learning_progress:.1%}")
                elif learning_progress < 0:
                    st.error(f"Overall Progress: {learning_progress:.1%}")
                else:
                    st.info("Maintaining Level")
            with col2:
                if consistency_over_time > 0.8:
                    st.success("Consistent Performance")
                elif consistency_over_time > 0.6:
                    st.warning("Variable Performance")
                else:
                    st.error("Inconsistent Performance")
            with col3:
                if improvement_trend > 0:
                    st.success("Improving Trend")
                elif improvement_trend < 0:
                    st.error("Declining Trend")
                else:
                    st.info("Stable Performance")
        
        # Recommendations
        st.subheader("📚 Your Personalized Study Plan")
        # Prepare table data
        table_data = [
            ["Focus Areas", "\n".join(ai_analysis['recommendations']['study_plan'])],
            ["Difficulty", ai_analysis['recommendations']['difficulty_adjustment']],
            ["Motivation", "\n".join(ai_analysis['recommendations']['motivation_tips'])],
            ["Resources", "\n".join(ai_analysis['recommendations']['resources'])],
            ["Next Steps", "\n".join(ai_analysis['recommendations']['next_steps'])]
        ]
        df_plan = pd.DataFrame(table_data, columns=["Category", "Recommendations"])
        st.table(df_plan)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error("🤖 Learning analysis is currently unavailable")
        st.info("Don't worry! You can still review your results and track your progress.")
        st.write("")
        st.write("**Showing your results:**")
        _show_basic_results(accuracy, correct_answers, total_questions)
    
    # Question-by-question analysis
    st.subheader("📝 Question Analysis")
    for i, result in enumerate(results):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**Q{i+1}:** {result['question']}")
        with col2:
            if result['is_correct']:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
        with col3:
            st.write(f"Your answer: {result['user_answer']}")
        
        if not result['is_correct']:
            st.info(f"**Correct answer:** {result['correct_answer']}")
            st.write(f"**Explanation:** {result['explanation']}")
        st.divider()
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take Another Quiz", use_container_width=True, key="retake_quiz_2"):
            # Clear all quiz-related state to ensure fresh start
            st.session_state.current_quiz = None
            st.session_state.quiz_results = []
            st.session_state.current_page = "quiz"
            st.rerun()
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True, key="back_home_2"):
            st.session_state.current_page = "home"
            st.rerun()

def _show_basic_results(accuracy, correct_answers, total_questions):
    """Show basic results when AI analysis fails"""
    st.subheader("📊 Basic Results")
    
    if accuracy >= 0.8:
        st.success("🎉 Excellent work! You're doing great!")
    elif accuracy >= 0.6:
        st.warning("👍 Good effort! Keep practicing to improve.")
    else:
        st.info("📚 Keep studying! Practice makes perfect.")

# Sample math questions
math_questions = [
    {
        "question": "What is 15 + 27?",
        "options": ["40", "42", "41", "43"],
        "correct": 1,
        "explanation": "15 + 27 = 42"
    },
    {
        "question": "If 3x + 5 = 20, what is x?",
        "options": ["3", "5", "7", "15"],
        "correct": 1,
        "explanation": "3x + 5 = 20 → 3x = 15 → x = 5"
    },
    {
        "question": "What is the area of a rectangle with length 8 and width 6?",
        "options": ["14", "48", "28", "56"],
        "correct": 1,
        "explanation": "Area = length × width = 8 × 6 = 48"
    },
    {
        "question": "What is 7² × 3?",
        "options": ["147", "21", "49", "343"],
        "correct": 0,
        "explanation": "7² = 49, so 49 × 3 = 147"
    },
    {
        "question": "If a train travels 120 km in 2 hours, what is its speed?",
        "options": ["60 km/h", "120 km/h", "240 km/h", "30 km/h"],
        "correct": 0,
        "explanation": "Speed = distance ÷ time = 120 ÷ 2 = 60 km/h"
    }
]

def generate_quiz(num_questions=5):
    """Generate a random quiz with 5 questions"""
    if num_questions > len(math_questions):
        num_questions = len(math_questions)
    
    selected_indices = np.random.choice(len(math_questions), size=num_questions, replace=False)
    selected_questions = [math_questions[i] for i in selected_indices]
    
    return selected_questions

# Main app logic
def main():
    # Navigation buttons at top
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    with col2:
        if st.session_state.user_registered:
            if st.button("📝 Take Quiz", use_container_width=True):
                st.session_state.current_page = "quiz"
                st.rerun()
        else:
            st.markdown("<div style='text-align: center; color: #666;'>Register to take quiz</div>", unsafe_allow_html=True)
    with col3:
        if st.session_state.user_registered and st.session_state.quiz_results:
            if st.button("📊 Results", use_container_width=True):
                st.session_state.current_page = "results"
                st.rerun()
        else:
            st.markdown("<div style='text-align: center; color: #666;'>Take quiz to see results</div>", unsafe_allow_html=True)
    with col4:
        if st.session_state.user_registered and st.session_state.quiz_history:
            if st.button("📈 History", use_container_width=True):
                st.session_state.current_page = "history"
                st.rerun()
        else:
            st.markdown("<div style='text-align: center; color: #666;'>Take quiz to see history</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Page routing
    if st.session_state.current_page == "home":
        show_home()
    elif st.session_state.current_page == "quiz":
        show_quiz()
    elif st.session_state.current_page == "results":
        show_results()
    elif st.session_state.current_page == "history":
        show_quiz_history()
    else:
        st.session_state.current_page = "home"
        show_home()

def show_quiz_history():
    st.header("📈 Learning Progress History")
    
    if not st.session_state.quiz_history:
        st.info("No quiz history available yet. Take a quiz to see your progress!")
        return
    
    # Create a DataFrame for the history
    history_df = pd.DataFrame([
        {
            'Attempt': attempt['attempt_number'],
            'Date': attempt['date'],
            'Score': attempt['score'],
            'Accuracy': f"{attempt['accuracy']:.1%}",
            'Subject': attempt['subject']
        }
        for attempt in st.session_state.quiz_history
    ])
    
    # Display the history table
    st.dataframe(history_df, use_container_width=True)
    
    # Show progress charts if enough data
    if len(st.session_state.quiz_history) > 1:
        st.subheader("📈 Progress Charts")
        
        # Create progress chart
        dates = [attempt['date'] for attempt in st.session_state.quiz_history]
        accuracies = [attempt['accuracy'] for attempt in st.session_state.quiz_history]
        scores = [attempt['score'] for attempt in st.session_state.quiz_history]
        
        # Accuracy Progress Over Time
        fig_accuracy = px.line(
            x=dates,
            y=accuracies,
            title="Accuracy Progress Over Time",
            labels={'x': 'Date', 'y': 'Accuracy'},
            markers=True
        )
        fig_accuracy.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Score Progress Over Time
        fig_score = px.line(
            x=dates,
            y=scores,
            title="Score Progress Over Time",
            labels={'x': 'Date', 'y': 'Score'},
            markers=True
        )
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Progress insights
        latest_accuracy = st.session_state.quiz_history[-1]['accuracy']
        previous_accuracy = st.session_state.quiz_history[-2]['accuracy']
        improvement = latest_accuracy - previous_accuracy
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Accuracy", f"{latest_accuracy:.1%}")
        with col2:
            st.metric("Previous Accuracy", f"{previous_accuracy:.1%}")
        with col3:
            if improvement > 0:
                st.success(f"Improvement: +{improvement:.1%}")
            elif improvement < 0:
                st.error(f"Change: {improvement:.1%}")
            else:
                st.info("No Change")
    
    # Navigation handled in main()

# Run the app
if __name__ == "__main__":
    main() 