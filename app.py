import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
import itertools
warnings.filterwarnings('ignore')

# Try to import BoTorch, fallback to scikit-learn if not available
try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.utils.transforms import normalize, unnormalize
    BOTORCH_AVAILABLE = True
    st.session_state.optimization_backend = "BoTorch"
except ImportError:
    BOTORCH_AVAILABLE = False
    st.session_state.optimization_backend = "Scikit-learn"

# Page config
st.set_page_config(
    page_title="Optimize Everything",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .suggestion-card {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'variables' not in st.session_state:
    st.session_state.variables = []
if 'objectives' not in st.session_state:
    st.session_state.objectives = []
if 'samples' not in st.session_state:
    st.session_state.samples = pd.DataFrame()
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []
if 'log_scaling_settings' not in st.session_state:
    st.session_state.log_scaling_settings = {
        'auto_log_scale': True,
        'manual_var_log_scale': {},
        'manual_obj_log_scale': {}
    }

def main():
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🎯 Optimize Everything</h1>
        <p>Multi-variable Bayesian optimization powered by {st.session_state.optimization_backend}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Variables section
        st.subheader("🎛️ Variables")
        with st.expander("Add Variables", expanded=len(st.session_state.variables) == 0):
            var_name = st.text_input("Variable name", key="var_name")
            col1, col2 = st.columns(2)
            with col1:
                var_min = st.number_input("Min value", key="var_min", format="%.4f", step=0.0001)
            with col2:
                var_max = st.number_input("Max value", key="var_max", format="%.4f", step=0.0001)
            
            if st.button("Add Variable"):
                add_variable(var_name, var_min, var_max)
        
        # Display current variables
        if st.session_state.variables:
            st.write("**Current Variables:**")
            for i, var in enumerate(st.session_state.variables):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {var['name']}: [{var['min']}, {var['max']}]")
                with col2:
                    if st.button("❌", key=f"del_var_{i}"):
                        remove_variable(i)
        
        # Objectives section
        st.subheader("🎯 Objectives")
        with st.expander("Add Objectives", expanded=len(st.session_state.objectives) == 0):
            obj_name = st.text_input("Objective name", key="obj_name")
            obj_type = st.selectbox("Type", ["maximize", "minimize"], key="obj_type")
            
            if st.button("Add Objective"):
                add_objective(obj_name, obj_type)
        
        # Display current objectives
        if st.session_state.objectives:
            st.write("**Current Objectives:**")
            for i, obj in enumerate(st.session_state.objectives):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {obj['name']} ({obj['type']})")
                with col2:
                    if st.button("❌", key=f"del_obj_{i}"):
                        remove_objective(i)
    
    # Main content
    if not st.session_state.variables or not st.session_state.objectives:
        st.warning("Please add at least one variable and one objective to get started!")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Input", "🎯 Optimization", "📈 Analysis", "🗺️ Contour Plots", "💾 Export/Import"])
    
    with tab1:
        data_input_section()
    
    with tab2:
        optimization_section()
    
    with tab3:
        analysis_section()
    
    with tab4:
        contour_plots_section()
    
    with tab5:
        export_import_section()

def add_variable(name, min_val, max_val):
    if not name:
        st.error("Please enter a variable name")
        return
    
    if any(var['name'] == name for var in st.session_state.variables):
        st.error("Variable name already exists")
        return
    
    if min_val >= max_val:
        st.error("Minimum value must be less than maximum value")
        return
    
    st.session_state.variables.append({
        'name': name,
        'min': min_val,
        'max': max_val
    })
    
    # Initialize log scaling setting for new variable
    st.session_state.log_scaling_settings['manual_var_log_scale'][name] = should_log_scale(min_val, max_val)
    
    st.success(f"Added variable: {name}")
    st.rerun()

def should_log_scale(min_val, max_val):
    """Determine if log scaling should be applied based on ratio and non-negativity"""
    if min_val <= 0:
        return False
    ratio = max_val / min_val
    return np.log10(ratio) > 2

def get_log_scaling_for_data(data, name, is_variable=True):
    """Get log scaling setting for a variable or objective"""
    if st.session_state.log_scaling_settings['auto_log_scale']:
        min_val, max_val = data.min(), data.max()
        return should_log_scale(min_val, max_val)
    else:
        if is_variable:
            return st.session_state.log_scaling_settings['manual_var_log_scale'].get(name, False)
        else:
            return st.session_state.log_scaling_settings['manual_obj_log_scale'].get(name, False)

def apply_log_scaling(data, should_scale):
    """Apply log scaling if needed"""
    if should_scale and (data > 0).all():
        return np.log10(data)
    return data

def reverse_log_scaling(data, was_scaled):
    """Reverse log scaling if it was applied"""
    if was_scaled:
        return 10 ** data
    return data
    removed_var = st.session_state.variables.pop(index)
    # Remove corresponding columns from samples
    if not st.session_state.samples.empty and removed_var['name'] in st.session_state.samples.columns:
        st.session_state.samples = st.session_state.samples.drop(columns=[removed_var['name']])
    st.rerun()

def add_objective(name, obj_type):
    if not name:
        st.error("Please enter an objective name")
        return
    
    if any(obj['name'] == name for obj in st.session_state.objectives):
        st.error("Objective name already exists")
        return
    
    st.session_state.objectives.append({
        'name': name,
        'type': obj_type
    })
    
    # Initialize log scaling setting for new objective
    st.session_state.log_scaling_settings['manual_obj_log_scale'][name] = False
    
    st.success(f"Added objective: {name}")
    st.rerun()

def remove_objective(index):
    removed_obj = st.session_state.objectives.pop(index)
    # Remove corresponding columns from samples
    if not st.session_state.samples.empty and removed_obj['name'] in st.session_state.samples.columns:
        st.session_state.samples = st.session_state.samples.drop(columns=[removed_obj['name']])
    st.rerun()

def data_input_section():
    st.header("📊 Current Samples")
    
    # Manual data entry
    st.subheader("Add New Sample")
    
    # Create input form
    with st.form("sample_form"):
        cols = st.columns(len(st.session_state.variables) + len(st.session_state.objectives))
        
        sample_data = {}
        
        # Variable inputs
        for i, var in enumerate(st.session_state.variables):
            with cols[i]:
                value = st.number_input(
                    f"{var['name']}", 
                    min_value=var['min'], 
                    max_value=var['max'],
                    value=(var['min'] + var['max']) / 2,
                    format="%.4f",
                    step=0.0001,
                    key=f"input_{var['name']}"
                )
                sample_data[var['name']] = value
        
        # Objective inputs
        for i, obj in enumerate(st.session_state.objectives):
            with cols[len(st.session_state.variables) + i]:
                value = st.number_input(
                    f"{obj['name']}", 
                    format="%.4f",
                    step=0.0001,
                    key=f"input_{obj['name']}"
                )
                sample_data[obj['name']] = value
        
        if st.form_submit_button("Add Sample"):
            add_sample(sample_data)
    
    # File upload
    st.subheader("Upload CSV Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            expected_columns = [var['name'] for var in st.session_state.variables] + [obj['name'] for obj in st.session_state.objectives]
            
            if all(col in df.columns for col in expected_columns):
                st.session_state.samples = pd.concat([st.session_state.samples, df[expected_columns]], ignore_index=True)
                st.success(f"Loaded {len(df)} samples from CSV")
                st.rerun()
            else:
                st.error(f"CSV must contain columns: {expected_columns}")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    # Display current samples
    if not st.session_state.samples.empty:
        st.subheader("Current Samples")
        
        # Create a dataframe with row indices for deletion
        display_df = st.session_state.samples.copy()
        display_df.index.name = 'Index'
        
        # Show samples with delete buttons
        for idx in display_df.index:
            col1, col2 = st.columns([6, 1])
            with col1:
                # Display row data
                row_data = []
                for col in display_df.columns:
                    row_data.append(f"{col}: {display_df.loc[idx, col]:.4f}")
                st.write(f"**Sample {idx}:** " + " | ".join(row_data))
            with col2:
                if st.button("🗑️", key=f"delete_sample_{idx}", help="Delete this sample"):
                    st.session_state.samples = st.session_state.samples.drop(idx).reset_index(drop=True)
                    st.rerun()
        
        # Summary and clear all button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Total samples: {len(st.session_state.samples)}")
        with col2:
            if st.button("Clear All Samples"):
                st.session_state.samples = pd.DataFrame()
                st.rerun()
    else:
        st.info("No samples added yet. Add some data to get started with optimization!")

def add_sample(sample_data):
    if st.session_state.samples.empty:
        st.session_state.samples = pd.DataFrame([sample_data])
    else:
        st.session_state.samples = pd.concat([st.session_state.samples, pd.DataFrame([sample_data])], ignore_index=True)
    st.success("Sample added successfully!")
    st.rerun()

def optimization_section():
    st.header("🎯 Bayesian Optimization")
    
    # Show backend info
    if BOTORCH_AVAILABLE:
        st.success("✅ Using BoTorch for advanced Bayesian optimization")
    else:
        st.warning("⚠️ BoTorch not available. Using scikit-learn Gaussian Process (still very effective!)")
    
    if st.session_state.samples.empty:
        st.warning("Please add some sample data first!")
        return
    
    if len(st.session_state.samples) < 2:
        st.warning("Need at least 2 samples for Bayesian optimization!")
        return
    
    # Optimization settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_suggestions = st.number_input("Number of suggestions", min_value=1, max_value=20, value=5)
    
    with col2:
        acquisition_func = st.selectbox("Acquisition Function", ["Expected Improvement", "Upper Confidence Bound"])
    
    with col3:
        if acquisition_func == "Upper Confidence Bound":
            beta = st.number_input("UCB Beta", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        else:
            beta = None
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        st.subheader("Log Scaling Options")
        
        auto_log_scale = st.checkbox(
            "Auto log-scale variables/objectives (when ratio > 100 and values > 0)",
            value=st.session_state.log_scaling_settings['auto_log_scale']
        )
        st.session_state.log_scaling_settings['auto_log_scale'] = auto_log_scale
        
        if not auto_log_scale:
            st.write("**Manual Variable Log Scaling:**")
            for var in st.session_state.variables:
                current_setting = st.session_state.log_scaling_settings['manual_var_log_scale'].get(var['name'], False)
                new_setting = st.checkbox(f"Log-scale {var['name']}", value=current_setting, key=f"log_var_{var['name']}")
                st.session_state.log_scaling_settings['manual_var_log_scale'][var['name']] = new_setting
            
            st.write("**Manual Objective Log Scaling:**")
            for obj in st.session_state.objectives:
                current_setting = st.session_state.log_scaling_settings['manual_obj_log_scale'].get(obj['name'], False)
                new_setting = st.checkbox(f"Log-scale {obj['name']}", value=current_setting, key=f"log_obj_{obj['name']}")
                st.session_state.log_scaling_settings['manual_obj_log_scale'][obj['name']] = new_setting
    
    # Objective selection for optimization (if multiple objectives)
    if len(st.session_state.objectives) > 1:
        selected_objective = st.selectbox(
            "Select objective to optimize", 
            [obj['name'] for obj in st.session_state.objectives]
        )
    else:
        selected_objective = st.session_state.objectives[0]['name']
    
    # Generate suggestions button
    if st.button("🚀 Generate Optimization Suggestions", type="primary"):
        with st.spinner("Running Bayesian optimization..."):
            try:
                # Show log scaling info
                if st.session_state.log_scaling_settings['auto_log_scale']:
                    st.info("ℹ️ Auto log-scaling enabled for variables/objectives with ratio > 100")
                
                suggestions = generate_bayesian_suggestions(
                    st.session_state.samples,
                    st.session_state.variables,
                    selected_objective,
                    num_suggestions,
                    acquisition_func,
                    beta
                )
                
                st.success("Optimization completed!")
                
                # Display suggestions
                st.subheader("💡 Recommended Experiments")
                
                for i, suggestion in enumerate(suggestions):
                    with st.container():
                        st.markdown(f"""
                        <div class="suggestion-card">
                            <h4>Experiment {i+1}</h4>
                            <p>{format_suggestion(suggestion, st.session_state.variables)}</p>
                            <small>Acquisition Value: {suggestion.get('acquisition_value', 'N/A'):.4f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Store in history
                st.session_state.optimization_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'objective': selected_objective,
                    'suggestions': suggestions,
                    'num_samples': len(st.session_state.samples),
                    'backend': st.session_state.optimization_backend
                })
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.exception(e)

def generate_bayesian_suggestions(samples_df, variables, objective_name, num_suggestions, acq_func, beta=None):
    """Generate optimization suggestions using BoTorch or scikit-learn fallback"""
    
    if BOTORCH_AVAILABLE:
        return generate_botorch_suggestions(samples_df, variables, objective_name, num_suggestions, acq_func, beta)
    else:
        return generate_sklearn_suggestions(samples_df, variables, objective_name, num_suggestions, acq_func, beta)

def generate_botorch_suggestions(samples_df, variables, objective_name, num_suggestions, acq_func, beta=None):
    """Generate optimization suggestions using BoTorch with log scaling"""
    
    # Prepare data with log scaling
    var_names = [var['name'] for var in variables]
    X_raw = samples_df[var_names].values.copy()
    y_raw = samples_df[objective_name].values.reshape(-1, 1).copy()
    
    # Apply log scaling to variables
    var_log_scaled = []
    for i, var in enumerate(variables):
        should_scale = get_log_scaling_for_data(samples_df[var['name']], var['name'], is_variable=True)
        var_log_scaled.append(should_scale)
        if should_scale:
            X_raw[:, i] = apply_log_scaling(X_raw[:, i], True)
    
    # Apply log scaling to objective
    obj_should_scale = get_log_scaling_for_data(samples_df[objective_name], objective_name, is_variable=False)
    if obj_should_scale:
        y_raw = apply_log_scaling(y_raw.flatten(), True).reshape(-1, 1)
    
    # Find if we should maximize or minimize
    obj_info = next(obj for obj in st.session_state.objectives if obj['name'] == objective_name)
    minimize = obj_info['type'] == 'minimize'
    
    # If minimizing, negate the objective
    if minimize:
        y_raw = -y_raw
    
    # Convert to torch tensors
    X = torch.tensor(X_raw, dtype=torch.float64)
    y = torch.tensor(y_raw, dtype=torch.float64)
    
    # Create bounds for normalization (considering log scaling)
    torch_bounds = []
    for i, var in enumerate(variables):
        if var_log_scaled[i]:
            min_val = np.log10(max(var['min'], 1e-10))
            max_val = np.log10(var['max'])
        else:
            min_val = var['min']
            max_val = var['max']
        torch_bounds.append([min_val, max_val])
    
    bounds = torch.tensor(torch_bounds, dtype=torch.float64).T
    X_normalized = normalize(X, bounds)
    
    # Create and fit GP model
    model = SingleTaskGP(X_normalized, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    # Define acquisition function
    if acq_func == "Expected Improvement":
        acquisition = ExpectedImprovement(model, best_f=y.max())
    else:  # Upper Confidence Bound
        acquisition = UpperConfidenceBound(model, beta=beta or 2.0)
    
    # Optimize acquisition function
    candidates, acq_values = optimize_acqf(
        acquisition,
        bounds=torch.stack([torch.zeros(len(variables)), torch.ones(len(variables))]),
        q=num_suggestions,
        num_restarts=20,
        raw_samples=512,
    )
    
    # Unnormalize candidates
    candidates_unnormalized = unnormalize(candidates, bounds)
    
    # Convert back to numpy and create suggestions
    suggestions = []
    for i in range(num_suggestions):
        suggestion = {}
        for j, var in enumerate(variables):
            value = float(candidates_unnormalized[i, j].item())
            # Reverse log scaling if applied
            if var_log_scaled[j]:
                value = 10 ** value
            suggestion[var['name']] = value
        suggestion['acquisition_value'] = float(acq_values[i].item())
        suggestions.append(suggestion)
    
    return suggestions

def generate_sklearn_suggestions(samples_df, variables, objective_name, num_suggestions, acq_func, beta=None):
    """Generate optimization suggestions using scikit-learn Gaussian Process with log scaling"""
    
    # Prepare data with log scaling
    var_names = [var['name'] for var in variables]
    X_raw = samples_df[var_names].values.copy()
    y_raw = samples_df[objective_name].values.copy()
    
    # Apply log scaling to variables
    var_log_scaled = []
    for i, var in enumerate(variables):
        should_scale = get_log_scaling_for_data(samples_df[var['name']], var['name'], is_variable=True)
        var_log_scaled.append(should_scale)
        if should_scale:
            X_raw[:, i] = apply_log_scaling(X_raw[:, i], True)
    
    # Apply log scaling to objective
    obj_should_scale = get_log_scaling_for_data(samples_df[objective_name], objective_name, is_variable=False)
    if obj_should_scale:
        y_raw = apply_log_scaling(y_raw, True)
    
    # Find if we should maximize or minimize
    obj_info = next(obj for obj in st.session_state.objectives if obj['name'] == objective_name)
    minimize = obj_info['type'] == 'minimize'
    
    # If minimizing, negate the objective
    if minimize:
        y_raw = -y_raw
    
    # Normalize inputs to [0, 1]
    X_normalized = np.zeros_like(X_raw)
    bounds = []
    for i, var in enumerate(variables):
        if var_log_scaled[i]:
            # Use log-scaled bounds
            min_val = np.log10(max(var['min'], 1e-10))
            max_val = np.log10(var['max'])
        else:
            min_val = var['min']
            max_val = var['max']
        
        X_normalized[:, i] = (X_raw[:, i] - min_val) / (max_val - min_val)
        bounds.append([0, 1])
    
    # Create and fit GP model
    kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X_normalized, y_raw)
    
    # Generate candidate points
    np.random.seed(42)  # For reproducibility
    n_candidates = 1000
    candidates = np.random.uniform(0, 1, (n_candidates, len(variables)))
    
    # Predict mean and std for candidates
    mean, std = gp.predict(candidates, return_std=True)
    
    # Calculate acquisition function
    if acq_func == "Expected Improvement":
        best_f = np.max(y_raw)
        z = (mean - best_f) / (std + 1e-9)
        ei = (mean - best_f) * norm_cdf(z) + std * norm_pdf(z)
        acq_values = ei
    else:  # Upper Confidence Bound
        beta = beta or 2.0
        acq_values = mean + beta * std
    
    # Select top suggestions
    top_indices = np.argsort(acq_values)[-num_suggestions:][::-1]
    
    suggestions = []
    for idx in top_indices:
        suggestion = {}
        # Unnormalize candidates
        for j, var in enumerate(variables):
            if var_log_scaled[j]:
                # Reverse log scaling
                min_val = np.log10(max(var['min'], 1e-10))
                max_val = np.log10(var['max'])
                unnormalized_log_value = candidates[idx, j] * (max_val - min_val) + min_val
                unnormalized_value = 10 ** unnormalized_log_value
            else:
                unnormalized_value = candidates[idx, j] * (var['max'] - var['min']) + var['min']
            
            suggestion[var['name']] = float(unnormalized_value)
        suggestion['acquisition_value'] = float(acq_values[idx])
        suggestions.append(suggestion)
    
    return suggestions-5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X_normalized, y_raw)
    
    # Generate candidate points
    np.random.seed(42)  # For reproducibility
    n_candidates = 1000
    candidates = np.random.uniform(0, 1, (n_candidates, len(variables)))
    
    # Predict mean and std for candidates
    mean, std = gp.predict(candidates, return_std=True)
    
    # Calculate acquisition function
    if acq_func == "Expected Improvement":
        best_f = np.max(y_raw)
        z = (mean - best_f) / (std + 1e-9)
        ei = (mean - best_f) * norm_cdf(z) + std * norm_pdf(z)
        acq_values = ei
    else:  # Upper Confidence Bound
        beta = beta or 2.0
        acq_values = mean + beta * std
    
    # Select top suggestions
    top_indices = np.argsort(acq_values)[-num_suggestions:][::-1]
    
    suggestions = []
    for idx in top_indices:
        suggestion = {}
        # Unnormalize candidates
        for j, var in enumerate(variables):
            unnormalized_value = candidates[idx, j] * (var['max'] - var['min']) + var['min']
            suggestion[var['name']] = float(unnormalized_value)
        suggestion['acquisition_value'] = float(acq_values[idx])
        suggestions.append(suggestion)
    
    return suggestions

def norm_cdf(x):
    """Standard normal CDF approximation"""
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def norm_pdf(x):
    """Standard normal PDF"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def format_suggestion(suggestion, variables):
    """Format suggestion for display with 4 decimal places"""
    var_strings = []
    for var in variables:
        value = suggestion[var['name']]
        var_strings.append(f"<strong>{var['name']}:</strong> {value:.4f}")
    return ", ".join(var_strings)

def analysis_section():
    st.header("📈 Analysis & Visualization")
    
    if st.session_state.samples.empty:
        st.info("Add some sample data to see analysis!")
        return
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.dataframe(st.session_state.samples.describe(), use_container_width=True)
    
    # Correlation matrix
    if len(st.session_state.samples.columns) > 1:
        st.subheader("🔗 Correlation Matrix")
        corr_matrix = st.session_state.samples.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Variable & Objective Correlations")
        st.plotly_chart(fig, use_container_width=True)
    
    # Pair plots for variables
    if len(st.session_state.variables) >= 2:
        st.subheader("🎯 Variable Relationships")
        var_names = [var['name'] for var in st.session_state.variables]
        
        if len(st.session_state.objectives) > 0:
            obj_name = st.selectbox("Color by objective:", [obj['name'] for obj in st.session_state.objectives])
            fig = px.scatter_matrix(st.session_state.samples, 
                                  dimensions=var_names,
                                  color=obj_name,
                                  title="Variable Space Exploration")
        else:
            fig = px.scatter_matrix(st.session_state.samples, 
                                  dimensions=var_names,
                                  title="Variable Space Exploration")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Objective trends
    if len(st.session_state.objectives) > 0:
        st.subheader("📈 Objective Trends")
        obj_cols = st.columns(len(st.session_state.objectives))
        
        for i, obj in enumerate(st.session_state.objectives):
            with obj_cols[i]:
                fig = px.line(y=st.session_state.samples[obj['name']], 
                             title=f"{obj['name']} Over Time",
                             labels={'index': 'Experiment #', 'y': obj['name']})
                fig.add_scatter(y=st.session_state.samples[obj['name']], mode='markers')
                st.plotly_chart(fig, use_container_width=True)
    
    # Optimization history
    if st.session_state.optimization_history:
        st.subheader("🕒 Optimization History")
        history_df = pd.DataFrame([
            {
                'Timestamp': hist['timestamp'],
                'Objective': hist['objective'],
                'Samples Used': hist['num_samples'],
                'Suggestions Generated': len(hist['suggestions'])
            }
            for hist in st.session_state.optimization_history
        ])
        st.dataframe(history_df, use_container_width=True)

def contour_plots_section():
    st.header("🗺️ Contour Plots")
    
    if st.session_state.samples.empty:
        st.info("Add some sample data to see contour plots!")
        return
    
    if len(st.session_state.variables) < 2:
        st.info("Need at least 2 variables to create contour plots!")
        return
    
    if len(st.session_state.objectives) == 0:
        st.info("Need at least 1 objective to create contour plots!")
        return
    
    # Get all variable combinations
    var_combinations = list(itertools.combinations(st.session_state.variables, 2))
    
    if len(var_combinations) == 0:
        st.info("Need at least 2 variables for contour plots!")
        return
    
    # Select objective for contour plotting
    selected_objective = st.selectbox(
        "Select objective for contour plots:",
        [obj['name'] for obj in st.session_state.objectives]
    )
    
    # Create contour plots for each variable combination
    n_plots = len(var_combinations)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    for i, (var1, var2) in enumerate(var_combinations):
        if i % n_cols == 0:
            cols = st.columns(n_cols)
        
        with cols[i % n_cols]:
            fig = create_contour_plot(
                st.session_state.samples,
                var1['name'], var2['name'], selected_objective
            )
            st.plotly_chart(fig, use_container_width=True)

def create_contour_plot(samples_df, var1_name, var2_name, obj_name):
    """Create a contour plot for two variables and one objective"""
    
    # Get data
    x = samples_df[var1_name].values
    y = samples_df[var2_name].values
    z = samples_df[obj_name].values
    
    # Create interpolation grid
    x_range = np.linspace(x.min(), x.max(), 50)
    y_range = np.linspace(y.min(), y.max(), 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Interpolate values for contour
    from scipy.interpolate import griddata
    Z_grid = griddata((x, y), z, (X_grid, Y_grid), method='cubic', fill_value=z.mean())
    
    # Create contour plot
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=Z_grid,
        colorscale='Viridis',
        name=obj_name,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(title=obj_name)
    ))
    
    # Add sample points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            color=z,
            colorscale='Viridis',
            line=dict(width=2, color='white'),
            showscale=False
        ),
        text=[f"{obj_name}: {val:.3f}" for val in z],
        hovertemplate=f"{var1_name}: %{{x:.3f}}<br>{var2_name}: %{{y:.3f}}<br>%{{text}}<extra></extra>",
        name="Samples"
    ))
    
    fig.update_layout(
        title=f"{obj_name} vs {var1_name} & {var2_name}",
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        height=400
    )
    
    return fig
    st.header("💾 Export & Import")
    
    # Export section
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.samples.empty:
            csv_data = st.session_state.samples.to_csv(index=False)
            st.download_button(
                label="Download Samples as CSV",
                data=csv_data,
                file_name="optimization_samples.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export complete configuration
        config_data = {
            'variables': st.session_state.variables,
            'objectives': st.session_state.objectives,
            'samples': st.session_state.samples.to_dict('records') if not st.session_state.samples.empty else [],
            'history': st.session_state.optimization_history
        }
        
        config_json = json.dumps(config_data, indent=2, default=str)
        st.download_button(
            label="Download Complete Configuration",
            data=config_json,
            file_name="optimization_config.json",
            mime="application/json"
        )
    
    # Import section
    st.subheader("📥 Import Configuration")
    
    uploaded_config = st.file_uploader("Upload Configuration JSON", type="json")
    if uploaded_config is not None:
        try:
            config_data = json.load(uploaded_config)
            
            if st.button("Load Configuration"):
                st.session_state.variables = config_data.get('variables', [])
                st.session_state.objectives = config_data.get('objectives', [])
                
                if config_data.get('samples'):
                    st.session_state.samples = pd.DataFrame(config_data['samples'])
                else:
                    st.session_state.samples = pd.DataFrame()
                
                st.session_state.optimization_history = config_data.get('history', [])
                
                st.success("Configuration loaded successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")

if __name__ == "__main__":
    main()
