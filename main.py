import streamlit as st
from optuna import load_study
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
import plotly.express as px

# Set the page config and theme
st.set_page_config(page_title="Optuna Dashboard", page_icon="🔍", layout="wide")

# Set custom colors for the plots
plotly_theme = {
    "plot_bgcolor": "rgba(0, 0, 0, 0)",
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
    "font": dict(family="Arial, sans-serif", size=14),
    "title": dict(font=dict(size=20)),
    "xaxis": dict(showgrid=True, gridcolor="lightgray", zeroline=False),
    "yaxis": dict(showgrid=True, gridcolor="lightgray", zeroline=False),
    "colorway": px.colors.qualitative.Safe,  # A vibrant color palette
}

# Title and description
st.title("Cytoplasm Segmentation Hyperparameter Optimization")

# Load the Optuna study
study = load_study(storage="sqlite:///optuna.db", study_name="cytoplasm_segmentation")

# Sidebar setup for navigation
sidebar = st.sidebar
sidebar.header("Navigation")
page = sidebar.radio("Select Page", ["Overview", "Trials", "Table"])

# Show overall study information and graphs
if page == "Overview":
    # Display general study info on the sidebar
    st.subheader("Study Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(label="Number of Trials", value=len(study.trials))
    col2.metric(label="Best Trial Value", value=round(study.best_trial.value, 3))
    col3.metric(label="Study Direction", value=study.direction.name.capitalize())

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Optimization History")
        opt_history = plot_optimization_history(study)
        opt_history.update_layout(plotly_theme)
        st.plotly_chart(opt_history, use_container_width=True)

        st.subheader("Hyperparameter Importance")
        param_importance = plot_param_importances(study)
        param_importance.update_layout(plotly_theme)
        st.plotly_chart(param_importance, use_container_width=True)

    with col2:
        st.subheader("Parallel Coordinate Plot")
        parallel_coord = plot_parallel_coordinate(study)
        parallel_coord.update_layout(plotly_theme)
        st.plotly_chart(parallel_coord, use_container_width=True)

        st.subheader("Slice Plot")
        slice_plot = plot_slice(study)
        slice_plot.update_layout(plotly_theme)
        st.plotly_chart(slice_plot, use_container_width=True)

elif page == "Trials":
    sidebar.subheader("Select Trial")
    trial_number = sidebar.selectbox(
        "Select Trial", [trial.number for trial in study.trials], label_visibility="collapsed"
    )

    selected_trial = study.trials[trial_number]
    st.subheader(f"Trial {selected_trial.number} Details")

    trial_info = {
        "Trial Number": selected_trial.number,
        "Value": selected_trial.value,
        "Parameters": selected_trial.params,
    }

    for key, value in trial_info.items():
        if isinstance(value, dict):
            st.write(f"**{key}:**")
            for sub_key, sub_value in value.items():
                st.metric(label=f"{sub_key}", value=sub_value)
        else:
            st.write(f"**{key}:** {value}")

elif page == "Table":
    st.subheader("Trials DataFrame")
    st.dataframe(study.trials_dataframe(), hide_index=True)
