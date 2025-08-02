# 🎓 Adaptive Learning System

A clean, modern adaptive learning system powered by real ASSISTments educational data.

## ✨ Features

- **Real Data**: Uses ASSISTments 2009-2010 dataset (1M+ records)
- **Clean Interface**: Beautiful magenta theme with simple navigation
- **User Registration**: Easy sign-up process
- **Adaptive Quiz**: 10-question mathematics quiz with difficulty distribution
- **Personalized Recommendations**: AI-powered learning suggestions
- **Analytics Dashboard**: Insights from real educational data

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run adaptive_learning_app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

## 📊 Data Processing

The system uses processed ASSISTments data:

- **7,920 learner profiles** extracted from real student data
- **35,965 questions** with difficulty classifications
- **1M+ performance records** for accurate analysis

## 🎯 How It Works

1. **Register**: Enter your name, email, and grade level
2. **Take Quiz**: Answer 10 mathematics questions
3. **Get Profile**: Receive your learner classification (Advanced/Moderate/Struggling)
4. **View Recommendations**: Get personalized learning suggestions
5. **Explore Analytics**: See insights from the ASSISTments dataset

## 🎨 Design

- **Clean magenta theme** with gradients
- **Responsive layout** for all devices
- **Intuitive navigation** with sidebar
- **Beautiful charts** using Plotly

## 📁 Project Structure

```
adaptive_learning_system/
├── adaptive_learning_app.py    # Main application
├── data/
│   ├── raw/                    # Raw ASSISTments data
│   ├── processed/              # Cleaned and processed data
│   └── assistments_processor.py # Data processing pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly
- **Data Source**: ASSISTments 2009-2010 Dataset

## 📈 Learner Types

The system classifies learners into three types:

- **Advanced**: High accuracy, fast completion
- **Moderate**: Balanced performance
- **Struggling**: Lower accuracy, needs support

## 🎓 Educational Impact

This system demonstrates how real educational data can be used to:
- Personalize learning experiences
- Identify student needs
- Provide targeted recommendations
- Improve educational outcomes

---

Built with ❤️ using real educational data from ASSISTments 