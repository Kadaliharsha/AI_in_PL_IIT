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

## 🧠 Model Training (Optional)

This project includes **two training systems**:

### **Basic Training** (`models/train.py`)
Simple training for two tasks:
- **interaction**: predict per-interaction correctness (binary)
- **learner**: predict per-student learner_type (multiclass)

### **Enhanced Training** (`models/enhanced_trainer.py`) ⭐ **RECOMMENDED**
Comprehensive training system with multiple specialized models:

1. **Learner Classification Model** - Predicts student type (Advanced/Moderate/Struggling)
2. **Performance Prediction Model** - Predicts success on next question  
3. **Engagement Analysis Model** - Predicts student engagement level

**Supported models**: MLP, RandomForest, SVM, GradientBoosting

### **Quick Start**

1) **Prepare data** (creates `data/processed/*.csv`):
```bash
python data/assistments_processor.py
```

2) **Train enhanced models** (recommended):
```bash
python models/enhanced_trainer.py
```

3) **Or train basic models**:
```bash
# Learner classification
python models/train.py --task learner --model mlp --epochs 100 --batch-size 32

# Interaction prediction  
python models/train.py --task interaction --model svm
```

### **Enhanced Training Features**
- ✅ **Multiple specialized models** for different learning aspects
- ✅ **Advanced feature engineering** (consistency, engagement, efficiency)
- ✅ **Comprehensive evaluation** (accuracy, CV scores, ROC AUC)
- ✅ **Cross-validation** for robust performance assessment
- ✅ **Automatic model saving** to `models/artifacts/`

### **Model Artifacts**
After training, you'll have models like:
- `learner_classification_mlp.pkl` - Student type prediction
- `performance_prediction_gb.pkl` - Question success prediction  
- `engagement_analysis_svm.pkl` - Engagement level analysis

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