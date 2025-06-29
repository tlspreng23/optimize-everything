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
                var_min = st.number_input("Min value", key="var_min")
            with col2:
                var_max = st.number_input("Max value", key="var_max")
            
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Input", "🎯 Optimization", "📈 Analysis", "💾 Export/Import"])
    
    with tab1:
        data_input_section()
    
    with tab2:
        optimization_section()
    
    with tab3:
        analysis_section()
    
    with tab4:
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
    st.success(f"Added variable: {name}")
    st.rerun()

def remove_variable(index):
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
                    key=f"input_{var['name']}"
                )
                sample_data[var['name']] = value
        
        # Objective inputs
        for i, obj in enumerate(st.session_state.objectives):
            with cols[len(st.session_state.variables) + i]:
                value = st.number_input(
                    f"{obj['name']}", 
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
        
        # Add delete functionality
        col1, col2 = st.columns([4, 1])
        with col1:
            st.dataframe(st.session_state.samples, use_container_width=True)
        with col2:
            if st.button("Clear All Samples"):
                st.session_state.samples = pd.DataFrame()
                st.rerun()
        
        st.info(f"Total samples: {len(st.session_state.samples)}")
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
                    'num_samples': len(st.session_state.samples)
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
    """Generate optimization suggestions using BoTorch"""
    
    # Prepare data
    var_names = [var['name'] for var in variables]
    X_raw = samples_df[var_names].values
    y_raw = samples_df[objective_name].values.reshape(-1, 1)
    
    # Find if we should maximize or minimize
    obj_info = next(obj for obj in st.session_state.objectives if obj['name'] == objective_name)
    minimize = obj_info['type'] == 'minimize'
    
    # If minimizing, negate the objective
    if minimize:
        y_raw = -y_raw
    
    # Convert to torch tensors
    X = torch.tensor(X_raw, dtype=torch.float64)
    y = torch.tensor(y_raw, dtype=torch.float64)
    
    # Normalize inputs to [0, 1]
    bounds = torch.tensor([[var['min'] for var in variables], 
                          [var['max'] for var in variables]], dtype=torch.float64)
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
            suggestion[var['name']] = float(candidates_unnormalized[i, j].item())
        suggestion['acquisition_value'] = float(acq_values[i].item())
        suggestions.append(suggestion)
    
    return suggestions

def generate_sklearn_suggestions(samples_df, variables, objective_name, num_suggestions, acq_func, beta=None):
    """Generate optimization suggestions using scikit-learn Gaussian Process"""
    
    # Prepare data
    var_names = [var['name'] for var in variables]
    X_raw = samples_df[var_names].values
    y_raw = samples_df[objective_name].values
    
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
        X_normalized[:, i] = (X_raw[:, i] - var['min']) / (var['max'] - var['min'])
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
    """Format suggestion for display"""
    var_strings = []
    for var in variables:
        value = suggestion[var['name']]
        var_strings.append(f"<strong>{var['name']}:</strong> {value:.3f}")
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

def export_import_section():
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
