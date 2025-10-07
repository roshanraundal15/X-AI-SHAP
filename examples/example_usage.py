"""
Example Usage of SHAP XAI Exploiter
Author: Roshan Raundal
Date: September 2025
Purpose: Demonstrate how to use the SHAP explainer library
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, load_diabetes
from sklearn.model_selection import train_test_split
from typing import Union, List, Optional, Dict, Any
import sys
import os
import shap
import joblib


# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from xai_exploiter.shap_explainer import SHAPExplainer
from xai_exploiter.visualizations import SHAPVisualizer



def safe_int_index(idx):
    if isinstance(idx, (np.ndarray, list)):
        if np.size(idx) == 1:
            return int(np.array(idx).item())
        else:
            return int(np.array(idx).flat[0])
    else:
        return int(idx)



def create_sample_classification_model():
    print("🎯 Creating sample classification model...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"✅ Model trained with accuracy: {model.score(X_test, y_test):.4f}")

    # Save model and data here
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sample_classification_model.pkl")
    df.drop('target', axis=1).to_csv('models/sample_data.csv', index=False)
    print("Model saved at models/sample_classification_model.pkl")
    print("Dataset saved at models/sample_data.csv")

    return model, df, feature_names



def create_sample_regression_model():
    print("📈 Creating sample regression model...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"✅ Model trained with R² score: {model.score(X_test, y_test):.4f}")
    return model, df, feature_names



def example_instance_explanation():
    print("\n" + "="*60)
    print("🔍 EXAMPLE 1: Individual Instance Explanation")
    print("="*60)
    model, data, feature_names = create_sample_classification_model()
    print("🧠 Initializing SHAP explainer...")
    explainer = SHAPExplainer(
        model=model,
        background_data=data.drop('target', axis=1),
        feature_names=feature_names,
        explainer_type="auto"
    )
    print("⚡ Explaining instance 0...")
    instance = data.drop('target', axis=1).iloc[0:1]
    explanation = explainer.explain_instance(instance)
    print(f"\n📊 Results for Instance 0:")
    print(f"Prediction: {explanation['prediction']:.4f}")
    print(f"Expected Value: {explanation['expected_value']:.4f}")
    print(f"Difference: {explanation['prediction'] - explanation['expected_value']:.4f}")
    shap_values = explanation['shap_values']
    top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
    print(f"\n🎯 Top 5 Contributing Features:")
    for i, idx in enumerate(top_indices):
        idx_int = safe_int_index(idx)
        value = shap_values[idx_int]
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.item()
            else:
                value = value.mean()
        contribution = "Positive" if value > 0 else "Negative"
        print(f"  {i+1}. {feature_names[idx_int]}: {value:.4f} ({contribution})")
    print("🎨 Creating visualizations...")
    visualizer = SHAPVisualizer()
    try:
        fig = visualizer.waterfall_plot(explanation, save_path="example_waterfall.png")
        print("📊 Waterfall plot saved as 'example_waterfall.png'")
    except Exception as e:
        print(f"⚠️ Could not create waterfall plot: {e}")
    return explainer, data



def example_global_analysis():
    print("\n" + "="*60)
    print("🌍 EXAMPLE 2: Global Model Analysis")
    print("="*60)
    model, data, feature_names = create_sample_regression_model()
    print("🧠 Initializing SHAP explainer...")
    explainer = SHAPExplainer(
        model=model,
        background_data=data.drop('target', axis=1),
        feature_names=feature_names,
        explainer_type="auto"
    )
    print("🔄 Performing global analysis...")
    analysis_data = data.drop('target', axis=1).sample(n=200, random_state=42)
    try:
        global_explanation = explainer.explain_global(analysis_data)
    except RuntimeError as e:
        print(f"⚠️ RuntimeError: {e}")
        print("🔄 Switching to KernelExplainer fallback...")
        import shap
        explainer.explainer = shap.KernelExplainer(explainer.model.predict, explainer.background_data)
        global_explanation = explainer.explain_global(analysis_data)


    importance_df = explainer.get_feature_importance(analysis_data)
    print(f"\n📈 Global Analysis Results:")
    print(f"Analyzed {len(analysis_data)} instances")
    print(f"Features analyzed: {len(feature_names)}")
    print(f"\n🏆 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:15s}: {row['importance']:.4f}")
    print("\n🎨 Creating global visualizations...")
    visualizer = SHAPVisualizer()
    try:
        fig = visualizer.feature_importance_plot(
            importance_df, 
            save_path="example_importance.png"
        )
        print("📊 Feature importance plot saved as 'example_importance.png'")
        visualizer.summary_plot(
            global_explanation['shap_values'],
            global_explanation['data'],
            global_explanation['feature_names'],
            save_path="example_summary.png"
        )
        print("📈 Summary plot saved as 'example_summary.png'")
    except Exception as e:
        print(f"⚠️ Could not create some plots: {e}")
    return explainer, data, importance_df



def main():
    print("🚀 XAI Exploiter SHAP Examples")
    print("Author: Roshan Raundal")
    print("Project: Python Library for BlackBox AI Exploitation")
    print("University: University of Mumbai")
    try:
        explainer1, data1 = example_instance_explanation()
        explainer2, data2, importance_df = example_global_analysis()
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📁 Generated Files:")
        print("  - example_waterfall.png")
        print("  - example_importance.png")
        print("  - example_summary.png")
        print("\n🎯 Next Steps:")
        print("  1. Try the CLI: python -m xai_exploiter.cli --help")
        print("  2. Launch dashboard: streamlit run dashboard/streamlit_app.py")
        print("  3. Explore the generated visualizations")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
