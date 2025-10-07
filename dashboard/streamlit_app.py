"""
Streamlit Dashboard for SHAP XAI Exploiter
Author: Roshan Raundal
Date: September 2025
Purpose: Interactive web dashboard for SHAP explanations
"""


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Optional, Any
import plotly.graph_objects as go
import shap


# Streamlit-SHAP integration
try:
    import streamlit_shap as st_shap
except ImportError:
    st.error("streamlit-shap not installed. Install with: pip install streamlit-shap")
    st.stop()


# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from xai_exploiter.shap_explainer import SHAPExplainer
    from xai_exploiter.visualizations import SHAPVisualizer
except ImportError:
    st.error("Could not import XAI Exploiter modules. Make sure the package is installed.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="XAI Exploiter - SHAP Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .author-info {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def load_data_and_model():
    """Load model and data from file paths or environment variables"""
    model_path = os.environ.get('SHAP_MODEL_PATH')
    data_path = os.environ.get('SHAP_DATA_PATH')


    if not model_path or not data_path:
        st.sidebar.header("📂 File Upload")


        model_file = st.sidebar.file_uploader(
            "Upload trained model (.pkl, .joblib)",
            type=['pkl', 'joblib'],
            help="Upload your trained model"
        )
        data_file = st.sidebar.file_uploader(
            "Upload dataset (.csv)",
            type=['csv'],
            help="Upload your dataset"
        )


        if model_file and data_file:
            # Use a relative directory inside your project, e.g., 'uploads/'
            upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
            os.makedirs(upload_dir, exist_ok=True)


            model_path = os.path.join(upload_dir, "uploaded_model.pkl")
            data_path = os.path.join(upload_dir, "uploaded_data.csv")


            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())


            with open(data_path, "wb") as f:
                f.write(data_file.getbuffer())
        else:
            return None, None, None, None


    try:
        model = joblib.load(model_path)
        data = pd.read_csv(data_path)
        return model, data, model_path, data_path


    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None, None


def initialize_explainer(model, data, explainer_type="auto"):
    """Initialize SHAP explainer with caching"""
    @st.cache_resource
    def _create_explainer(_model, _data, _explainer_type):
        return SHAPExplainer(
            model=_model,
            background_data=_data.sample(min(100, len(_data))),
            feature_names=_data.columns.tolist(),
            explainer_type=_explainer_type
        )
    return _create_explainer(model, data, explainer_type)


def main():
    """Main dashboard function"""


    # Header
    st.markdown('<h1 class="main-header">🔍 XAI Exploiter - SHAP Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="author-info">Developed by: Roshan Raundal | University of Mumbai</p>', unsafe_allow_html=True)


    # Load model and data
    model, data, model_path, data_path = load_data_and_model()


    if model is None or data is None:
        st.info("👆 Please upload your model and dataset files using the sidebar.")
        st.markdown("""
        ### 📋 Instructions:
        1. **Upload Model**: Upload your trained model file (.pkl or .joblib)
        2. **Upload Data**: Upload your dataset (.csv)
        3. **Explore**: Use the dashboard features to analyze your model


        ### 🎯 Supported Models:
        - Scikit-learn models (RandomForest, SVM, etc.)
        - XGBoost models
        - LightGBM models
        - Any model with `.predict()` method
        """)
        return


    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")


    # Dataset info
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Rows**: {len(data):,}")
    st.sidebar.write(f"**Columns**: {len(data.columns):,}")
    st.sidebar.write(f"**Features**: {', '.join(data.columns[:3])}{'...' if len(data.columns) > 3 else ''}")


    # Explainer configuration
    explainer_type = st.sidebar.selectbox(
        "SHAP Explainer Type",
        ["auto", "tree", "kernel", "explainer"],
        help="auto: Automatically select best explainer based on model type"
    )


    # Initialize explainer
    with st.spinner("🧠 Initializing SHAP explainer..."):
        explainer = initialize_explainer(model, data, explainer_type)


    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Instance Explanation", "🌍 Global Analysis", "📊 Feature Analysis", "📈 Model Insights"])


    # Tab 1: Instance Explanation
    with tab1:
        st.markdown('<h2 class="sub-header">Individual Instance Explanation</h2>', unsafe_allow_html=True)


        # Instance selection
        col1, col2 = st.columns([3, 1])


        with col1:
            instance_idx = st.selectbox(
                "Select instance to explain",
                options=range(min(100, len(data))),
                format_func=lambda x: f"Instance {x}",
                help="Choose an instance from the dataset to explain"
            )


        with col2:
            if st.button("🔍 Explain Instance", type="primary"):
                with st.spinner("⚡ Generating explanation..."):
                    try:
                        instance = data.iloc[instance_idx:instance_idx+1]
                        explanation = explainer.explain_instance(instance)


                        # Store explanation in session state
                        st.session_state.explanation = explanation
                        st.session_state.instance_idx = instance_idx
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
                        return


        # Display explanation if available
        if hasattr(st.session_state, 'explanation'):
            exp = st.session_state.explanation


            # Metrics
            col1, col2, col3 = st.columns(3)


            with col1:
                st.metric(
                    "Model Prediction", 
                    f"{exp['prediction']:.4f}",
                    help="The model's prediction for this instance"
                )


            with col2:
                st.metric(
                    "Base Value", 
                    f"{exp['expected_value']:.4f}",
                    help="The average prediction across the dataset"
                )


            with col3:
                diff = exp['prediction'] - exp['expected_value']
                st.metric(
                    "Difference from Base", 
                    f"{diff:.4f}",
                    delta=f"{diff:.4f}",
                    help="How much this prediction differs from the average"
                )


            # Visualizations
            st.subheader("📊 SHAP Visualizations")


            # Feature contributions table
            if exp['feature_names']:
                contrib_df = pd.DataFrame({
                    'Feature': exp['feature_names'],
                    'Value': data.iloc[st.session_state.instance_idx].values,
                    'SHAP Value': [x.mean() if hasattr(x, 'mean') else x for x in exp['shap_values']],
                    'Contribution': ['Positive' if (x.mean() if hasattr(x, 'mean') else x) > 0 else 'Negative' for x in exp['shap_values']]
                }).sort_values('SHAP Value', key=abs, ascending=False)


                st.dataframe(
                    contrib_df.style.format({'Value': '{:.4f}', 'SHAP Value': '{:.4f}'}),
                    use_container_width=True
                )


            # Waterfall plot
            visualizer = SHAPVisualizer()
            try:
                fig = visualizer.waterfall_plot(exp, title=f"SHAP Explanation - Instance {st.session_state.instance_idx}")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not create waterfall plot: {str(e)}")


    # Tab 2: Global Analysis
    with tab2:
        st.markdown('<h2 class="sub-header">Global Model Analysis</h2>', unsafe_allow_html=True)


        max_samples = st.slider(
            "Maximum samples to analyze",
            min_value=100,
            max_value=min(2000, len(data)),
            value=500,
            help="More samples = better analysis but slower computation"
        )


        if st.button("🌍 Run Global Analysis", type="primary"):
            with st.spinner("🔄 Computing SHAP values for all instances..."):
                try:
                    analysis_data = data.sample(n=min(max_samples, len(data)), random_state=42)


                    global_exp = explainer.explain_global(analysis_data)


                    st.session_state.global_exp = global_exp
                    st.session_state.analysis_data = analysis_data


                    st.success(f"✅ Analysis completed for {len(analysis_data)} instances!")


                except Exception as e:
                    st.error(f"Error in global analysis: {str(e)}")


        if hasattr(st.session_state, 'global_exp'):
            global_exp = st.session_state.global_exp


            # Feature importance
            st.subheader("🎯 Feature Importance")


            importance_values = global_exp['feature_importance']


            if hasattr(importance_values, 'ndim') and importance_values.ndim > 1:
                importance_values = importance_values.mean(axis=0)


            importance_values = importance_values.flatten()
            feature_names = global_exp['feature_names']


            min_len = min(len(feature_names), len(importance_values))
            feature_names = feature_names[:min_len]
            importance_values = importance_values[:min_len]


            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            }).sort_values('Importance', ascending=False)


            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=importance_df['Feature'][:15],
                x=importance_df['Importance'][:15],
                orientation='h',
                marker_color='steelblue'
            ))
            fig.update_layout(
                title="Top 15 Most Important Features",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)


            # Summary plot using streamlit-shap
            st.subheader("📈 SHAP Summary Plot")
            try:
                shap_values = global_exp['shap_values']


                # For multi-class shap_values, select class 1 if needed
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1]


                st_shap.st_shap(
                    shap.summary_plot(
                        shap_values, 
                        st.session_state.analysis_data,
                        feature_names=global_exp['feature_names'],
                        show=False
                    ),
                    height=400
                )
            except Exception as e:
                st.warning(f"Could not create summary plot: {str(e)}")


    # Tab 3: Feature Analysis
    with tab3:
        st.markdown('<h2 class="sub-header">Individual Feature Analysis</h2>', unsafe_allow_html=True)


        if hasattr(st.session_state, 'global_exp'):
            global_exp = st.session_state.global_exp


            selected_feature = st.selectbox(
                "Select feature for detailed analysis",
                options=global_exp['feature_names'],
                help="Choose a feature to analyze its relationship with SHAP values"
            )


            if selected_feature:
                feature_idx = global_exp['feature_names'].index(selected_feature)


                col1, col2, col3, col4 = st.columns(4)


                feature_values = st.session_state.analysis_data[selected_feature]


                with col1:
                    st.metric("Mean", f"{feature_values.mean():.4f}")
                with col2:
                    st.metric("Std", f"{feature_values.std():.4f}")
                with col3:
                    st.metric("Min", f"{feature_values.min():.4f}")
                with col4:
                    st.metric("Max", f"{feature_values.max():.4f}")


                st.subheader(f"📊 Dependence Plot: {selected_feature}")


                try:
                    shap_values_feature = global_exp['shap_values'][:, feature_idx]


                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=feature_values,
                        y=shap_values_feature,
                        mode='markers',
                        marker=dict(
                            color=shap_values_feature,
                            colorscale='RdBu',
                            showscale=True,
                            colorbar=dict(title="SHAP Value")
                        ),
                        name=selected_feature
                    ))


                    fig.update_layout(
                        title=f"SHAP Dependence Plot: {selected_feature}",
                        xaxis_title=f"{selected_feature} Value",
                        yaxis_title="SHAP Value",
                        height=500
                    )


                    st.plotly_chart(fig, use_container_width=True)


                except Exception as e:
                    st.error(f"Error creating dependence plot: {str(e)}")
        else:
            st.info("👈 Run global analysis first to see feature analysis.")


    # Tab 4: Model Insights
    with tab4:
        st.markdown('<h2 class="sub-header">Model Insights & Statistics</h2>', unsafe_allow_html=True)


        col1, col2 = st.columns(2)


        with col1:
            st.subheader("🤖 Model Information")
            st.write(f"**Model Type**: {type(model).__name__}")
            st.write(f"**Features**: {len(data.columns)}")
            st.write(f"**Dataset Size**: {len(data):,} rows")


            if hasattr(model, 'get_params'):
                params = model.get_params()
                st.write("**Key Parameters**:")
                for key, value in list(params.items())[:5]:
                    st.write(f"- {key}: {value}")


        with col2:
            st.subheader("📊 Dataset Overview")


            st.write("**Data Types**:")
            dtype_counts = data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")


            missing_values = data.isnull().sum().sum()
            st.write(f"**Missing Values**: {missing_values:,}")


        if hasattr(st.session_state, 'global_exp'):
            st.subheader("📈 SHAP Analysis Summary")


            global_exp = st.session_state.global_exp
            shap_values = global_exp['shap_values']


            col1, col2, col3 = st.columns(3)


            with col1:
                st.metric(
                    "Mean |SHAP|",
                    f"{np.abs(shap_values).mean():.4f}",
                    help="Average absolute SHAP value across all features and instances"
                )


            with col2:
                st.metric(
                    "SHAP Variance",
                    f"{shap_values.var():.4f}",
                    help="Variance in SHAP values indicates model complexity"
                )


            with col3:
                # Fix length alignment before computing max impact feature
                importance_values = global_exp['feature_importance']

                if hasattr(importance_values, 'ndim') and importance_values.ndim > 1:
                    importance_values = importance_values.mean(axis=0)

                importance_values = importance_values.flatten()
                feature_names = global_exp['feature_names']

                min_len = min(len(feature_names), len(importance_values))
                feature_names = feature_names[:min_len]
                importance_values = importance_values[:min_len]

                max_index = np.argmax(importance_values)
                max_impact_feature = feature_names[max_index]

                st.metric(
                    "Top Feature",
                    max_impact_feature,
                    help="Feature with highest average impact"
                )


            st.subheader("💾 Export Results")


            col1, col2, col3 = st.columns(3)


            with col1:
                if st.button("📄 Export Feature Importance"):
                    importance_values = global_exp['feature_importance']
                    if hasattr(importance_values, 'ndim') and importance_values.ndim > 1:
                        importance_values = importance_values.mean(axis=0)
                    importance_values = importance_values.flatten()
                    feature_names = global_exp['feature_names']
                    min_len = min(len(feature_names), len(importance_values))
                    feature_names = feature_names[:min_len]
                    importance_values = importance_values[:min_len]


                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance_values
                    }).sort_values('Importance', ascending=False)


                    csv = importance_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name="shap_feature_importance.csv",
                        mime="text/csv"
                    )


            with col2:
                if st.button("📊 Export SHAP Values"):
                    shap_vals = global_exp['shap_values']


                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]


                    shap_df = pd.DataFrame(
                        shap_vals,
                        columns=[f"SHAP_{col}" for col in global_exp['feature_names']]
                    )


                    csv = shap_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name="shap_values.csv",
                        mime="text/csv"
                    )


            with col3:
                if st.button("📋 Generate Report"):
                    report = f"""
# SHAP Analysis Report
Generated by: XAI Exploiter - Roshan Raundal


## Model Information
- Model Type: {type(model).__name__}
- Dataset Size: {len(data):,} rows, {len(data.columns)} features
- Analysis Samples: {len(st.session_state.analysis_data):,}


## Feature Importance (Top 10)
"""
                    importance_values = global_exp['feature_importance']
                    if hasattr(importance_values, 'ndim') and importance_values.ndim > 1:
                        importance_values = importance_values.mean(axis=0)
                    importance_values = importance_values.flatten()
                    feature_names = global_exp['feature_names']
                    min_len = min(len(feature_names), len(importance_values))
                    feature_names = feature_names[:min_len]
                    importance_values = importance_values[:min_len]


                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance_values
                    }).sort_values('Importance', ascending=False)


                    for i, row in importance_df.head(10).iterrows():
                        report += f"{row['Feature']}: {row['Importance']:.4f}\n"


                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name="shap_analysis_report.md",
                        mime="text/markdown"
                    )


if __name__ == "__main__":
    main()
