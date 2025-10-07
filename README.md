# XAI Exploiter - SHAP Component 🔍

**Author**: Roshan Raundal  
**University**: University of Mumbai  
**Project**: Python Library for BlackBox AI Exploitation  
**Component**: SHAP-based Explainability Module  

## 📋 Overview

This is the SHAP (SHapley Additive exPlanations) component of the larger XAI Exploiter library, designed to provide comprehensive model interpretability and transparency for machine learning models. This component specifically addresses the black-box nature of AI/ML models through theoretically sound explanations.

## 🎯 Features

### Core Functionality
- **Model-agnostic explanations** for any ML model
- **Automatic explainer selection** (TreeExplainer, KernelExplainer, General Explainer)
- **Individual instance explanations** with detailed feature contributions
- **Global model analysis** with feature importance rankings
- **Batch processing** for large datasets

### Visualization Capabilities
- 📊 **Waterfall plots** for instance-level explanations
- 📈 **Summary plots** (beeswarm/bar charts) for global insights  
- 🎯 **Feature importance plots** with interactive options
- 🔄 **Dependence plots** showing feature relationships
- 📋 **Decision plots** for prediction pathways

### Interface Options
- 🖥️ **Command-line interface (CLI)** for automation
- 🌐 **Interactive Streamlit dashboard** for exploration
- 🐍 **Python API** for programmatic access
- 📁 **Export capabilities** (JSON, HTML, PNG, CSV)

## 🚀 Quick Start

### Installation

```bash
# Clone the project
git clone <your-repo-url>
cd xai_exploiter_shap

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from xai_exploiter import SHAPExplainer, SHAPVisualizer
import pandas as pd
import joblib

# Load your model and data
model = joblib.load('your_model.pkl')
data = pd.read_csv('your_data.csv')

# Initialize explainer
explainer = SHAPExplainer(
    model=model,
    background_data=data,
    feature_names=data.columns.tolist()
)

# Explain a single instance
instance = data.iloc[0:1]
explanation = explainer.explain_instance(instance)

print(f"Prediction: {explanation['prediction']:.4f}")
print(f"Expected Value: {explanation['expected_value']:.4f}")

# Create visualizations
visualizer = SHAPVisualizer()
fig = visualizer.waterfall_plot(explanation)

# Global analysis
global_explanation = explainer.explain_global(data)
importance_df = explainer.get_feature_importance(data)
```

## 🖥️ CLI Usage

### Individual Instance Explanation
```bash
shap-xai explain-instance \
    --model-path model.pkl \
    --data-path data.csv \
    --instance-idx 5 \
    --output-dir ./results
```

### Global Model Analysis
```bash
shap-xai analyze-global \
    --model-path model.pkl \
    --data-path data.csv \
    --max-samples 1000 \
    --generate-report
```

### Interactive Dashboard
```bash
shap-xai dashboard \
    --model-path model.pkl \
    --data-path data.csv \
    --port 8501
```

### Feature Dependence Analysis
```bash
shap-xai dependence \
    --model-path model.pkl \
    --data-path data.csv \
    --feature-idx 2 \
    --output-path dependence_plot.png
```

### Get Help
```bash
shap-xai --help
shap-xai explain-instance --help
```

## 🌐 Dashboard Features

Launch the Streamlit dashboard to explore your model interactively:

- **📊 Instance Explanation**: Analyze individual predictions with interactive plots
- **🌍 Global Analysis**: Understand overall model behavior and feature importance
- **🔍 Feature Analysis**: Deep dive into individual feature contributions
- **📈 Model Insights**: Get comprehensive model statistics and export options

## 📁 Project Structure

```
xai_exploiter_shap/
├── xai_exploiter/
│   ├── __init__.py              # Package initialization
│   ├── shap_explainer.py        # Core SHAP functionality
│   ├── visualizations.py        # Visualization components
│   ├── cli.py                   # Command-line interface
│   └── utils.py                 # Utility functions
├── dashboard/
│   ├── streamlit_app.py         # Streamlit dashboard
│   └── components.py            # Dashboard components
├── examples/
│   ├── example_usage.py         # Usage examples
│   └── sample_data/             # Sample datasets
├── tests/
│   └── test_shap.py            # Unit tests
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## 🔧 Configuration Options

### SHAP Explainer Types

- **`auto`**: Automatically selects the best explainer
- **`tree`**: Use TreeExplainer for tree-based models (fastest)
- **`kernel`**: Use KernelExplainer for any model (slower but universal)
- **`explainer`**: Use the general SHAP Explainer

### Supported Models

- ✅ Scikit-learn models (RandomForest, SVM, etc.)
- ✅ XGBoost models
- ✅ LightGBM models  
- ✅ CatBoost models
- ✅ Any model with `.predict()` or `.predict_proba()` methods

## 📊 Examples

### Example 1: Classification Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xai_exploiter import SHAPExplainer

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Explain with SHAP
explainer = SHAPExplainer(model, X)
explanation = explainer.explain_instance(X[0:1])
```

### Example 2: Regression Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from xai_exploiter import SHAPExplainer, SHAPVisualizer

# Load data and train model
diabetes = load_diabetes()
model = RandomForestRegressor(random_state=42)
model.fit(diabetes.data, diabetes.target)

# Global analysis
explainer = SHAPExplainer(model, diabetes.data, diabetes.feature_names)
global_exp = explainer.explain_global(diabetes.data[:200])

# Visualizations
visualizer = SHAPVisualizer()
importance_df = explainer.get_feature_importance()
visualizer.feature_importance_plot(importance_df)
```

## 🧪 Testing

Run the example script to test the installation:

```bash
python examples/example_usage.py
```

This will:
- Create sample models
- Generate explanations
- Create visualizations
- Save results to files

## 🐛 Troubleshooting

### Common Issues

**ImportError for SHAP or Streamlit**:
```bash
pip install shap streamlit streamlit-shap
```

**Matplotlib backend issues**:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

**Large dataset performance**:
- Use `max_background_size` parameter to limit background data
- Sample your dataset before analysis
- Use TreeExplainer for tree models (much faster)

### Error Messages

**"Background data is required"**: Provide background data for KernelExplainer
**"Model not supported"**: Ensure your model has a `.predict()` method
**"Feature names mismatch"**: Check that feature names match your data columns

## 📈 Performance Tips

1. **Use TreeExplainer** for tree-based models (XGBoost, RandomForest)
2. **Limit background data** to 100-500 samples for faster computation
3. **Sample large datasets** before global analysis
4. **Use batch processing** for multiple instances
5. **Cache explanations** for repeated analysis

## 🤝 Integration with Team Components

This SHAP component is designed to integrate with:

- **Base Component**: Shared data preprocessing
- **LIME Component**: Complementary local explanations  
- **Counterfactual Components**: Alternative explanation methods
- **RL Component**: Advanced counterfactual generation

## 📚 References & Theory

SHAP is based on cooperative game theory and provides:

- **Efficiency**: Feature contributions sum to prediction difference
- **Symmetry**: Equal contribution for equal feature importance  
- **Dummy**: Zero contribution for irrelevant features
- **Additivity**: Consistent across different model combinations

Key papers:
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- Lundberg et al. (2020): "From local explanations to global understanding"

## 🎓 Academic Context

**University**: University of Mumbai  
**Degree**: BE Computer Engineering  
**Semester**: 7-8 (Major Project)  
**Guide**: Prof. D. S. Khachane  

**SDG Alignment**:
- **Goal 9**: Industry, Innovation, and Infrastructure
- **Goal 16**: Peace, Justice and Strong Institutions

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **SHAP Library**: Scott Lundberg and team
- **Streamlit**: For the amazing dashboard framework
- **University of Mumbai**: For project guidance and support
- **Team Members**: Karan Panchal, Faizal Shaikh, Soham Parag Pethkar

---

**Roshan Raundal** | University of Mumbai | 2025  
*"Making AI Explainable, One SHAP Value at a Time"* 🚀