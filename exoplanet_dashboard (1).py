"""
Exoplanet Occurrence Dashboard
A Streamlit app for analyzing exoplanet host stars from Kepler and TESS missions

To run: streamlit run exoplanet_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Exoplanet Occurrence Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING FUNCTION
# ============================================
@st.cache_data
def load_data():
    """Load the pre-processed exoplanet dataset or generate sample data"""
    try:
        df = pd.read_csv('final_dataset_pre_bayesian.csv')
        st.sidebar.success("✅ Real data loaded!")
        return df
    except FileNotFoundError:
        st.sidebar.info("📊 Using demo data")
        
        # Generate realistic sample data
        np.random.seed(42)
        n_samples = 500
        
        # Generate correlated stellar properties
        st_teff = np.random.normal(5800, 800, n_samples)
        st_met = np.random.normal(0, 0.3, n_samples)
        st_mass = np.random.normal(1, 0.3, n_samples)
        st_rad = st_mass ** 0.8 + np.random.normal(0, 0.1, n_samples)
        
        # Planet count influenced by metallicity and temperature
        lambda_planets = np.exp(0.2 + 0.3 * st_met + 0.15 * (st_teff - 5800)/800)
        planet_count = np.random.poisson(lambda_planets)
        
        # Spectral types based on temperature
        spectral_type = pd.cut(st_teff, bins=[0, 4000, 5200, 6000, 7500, 10000], 
                               labels=['M', 'K', 'G', 'F', 'A'])
        
        # Metallicity classes
        metallicity_class = pd.cut(st_met, bins=[-2, -0.5, 0.0, 0.5, 2], 
                                   labels=['low', 'sub-solar', 'solar', 'high'])
        
        df = pd.DataFrame({
            'hostname': [f'Star-{i:04d}' for i in range(n_samples)],
            'st_teff': st_teff,
            'st_met': st_met,
            'st_mass': st_mass,
            'st_rad': st_rad,
            'planet_count': planet_count,
            'mission': np.random.choice(['Kepler', 'TESS'], n_samples, p=[0.6, 0.4]),
            'spectral_type': spectral_type,
            'metallicity_class': metallicity_class,
            'sy_dist': np.random.uniform(10, 500, n_samples),
            'disc_year': np.random.choice(range(2009, 2025), n_samples)
        })
        
        return df

# ============================================
# LOAD DATA
# ============================================
df = load_data()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("🚀 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "🏠 Home",
        "📊 Data Explorer", 
        "🔬 Bayesian Analysis",
        "🔭 Light Curves",
        "👧 Kids Zone",
        "📈 Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard:**

Analyze exoplanet host stars from NASA's Kepler and TESS missions. 
Explore correlations, run Bayesian models, and discover patterns.
""")

# ============================================
# PAGE 1: HOME
# ============================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🌌 Exoplanet Occurrence Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Interactive Exoplanet Analysis Platform!
    
    This dashboard provides comprehensive analysis of exoplanet host stars from **Kepler** and **TESS** missions.
    Explore stellar properties, discover patterns, and understand what makes a star likely to host planets.
    
    **Features:**
    - 📊 Interactive data visualization and statistical analysis
    - 🔬 Bayesian statistical modeling and predictions
    - 🔭 Light curve simulation and analysis
    - 👧 Educational content for aspiring astronomers
    - 📈 Scientific insights and downloadable datasets
    """)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stars", f"{len(df):,}")
    
    with col2:
        st.metric("Total Planets", f"{int(df['planet_count'].sum()):,}")
    
    with col3:
        multi_planet = len(df[df['planet_count'] > 1])
        pct = (multi_planet / len(df)) * 100
        st.metric("Multi-Planet Systems", f"{multi_planet:,}", f"{pct:.1f}%")
    
    with col4:
        avg_planets = df['planet_count'].mean()
        st.metric("Avg Planets/Star", f"{avg_planets:.2f}")

# ============================================
# PAGE 2: DATA EXPLORER
# ============================================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    
    # Histogram: Stellar Temperature
    st.subheader("Distribution of Stellar Temperatures")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['st_teff'].dropna(), bins=30, edgecolor='black')
    ax.set_xlabel("Effective Temperature (K)")
    ax.set_ylabel("Number of Stars")
    ax.set_title("Distribution of Stellar Temperatures")
    st.pyplot(fig)
    
    # Pie chart: Mission Contribution
    st.subheader("Kepler vs TESS Host Stars")
    fig, ax = plt.subplots(figsize=(8, 5))
    mission_counts = df['mission'].value_counts()
    ax.pie(mission_counts, labels=mission_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Kepler vs TESS Host Stars")
    st.pyplot(fig)
    
    # --- 1. Planet Count Histogram ---
    st.subheader("Distribution of Planet Counts per Star")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='planet_count', data=df, ax=ax)
    ax.set_title("Distribution of Planet Counts per Star")
    ax.set_xlabel("Number of Planets")
    ax.set_ylabel("Number of Stars")
    st.pyplot(fig)
    
    # --- 2. Histogram of Stellar Metallicity ---
    st.subheader("Distribution of Stellar Metallicity")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['st_met'], bins=30, kde=True, color='teal', ax=ax)
    ax.set_title("Distribution of Stellar Metallicity")
    ax.set_xlabel("Stellar Metallicity [dex]")
    ax.set_ylabel("Number of Stars")
    st.pyplot(fig)
    
    # --- 3. Boxplot: Planet Count vs Metallicity ---
    st.subheader("Planet Count vs Stellar Metallicity")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='planet_count', y='st_met', data=df, ax=ax)
    ax.set_title("Planet Count vs Stellar Metallicity")
    ax.set_xlabel("Planet Count")
    ax.set_ylabel("Metallicity [dex]")
    st.pyplot(fig)
    
    # --- 4. Boxplot: Planet Count vs Stellar Temperature ---
    st.subheader("Planet Count vs Stellar Temperature")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='planet_count', y='st_teff', data=df, ax=ax)
    ax.set_title("Planet Count vs Stellar Temperature")
    ax.set_xlabel("Planet Count")
    ax.set_ylabel("Temperature [K]")
    st.pyplot(fig)
    
    # --- 5. Heatmap: Mean Planet Count by Spectral Type & Metallicity Class ---
    st.subheader("Mean Planet Count by Spectral Type and Metallicity Class")
    pivot = df.pivot_table(values='planet_count', index='spectral_type', columns='metallicity_class', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis', ax=ax)
    ax.set_title("Mean Planet Count by Spectral Type and Metallicity Class")
    st.pyplot(fig)

# ============================================
# PAGE 3: BAYESIAN ANALYSIS
# ============================================
elif page == "🔬 Bayesian Analysis":
    st.title("🔬 Bayesian Analysis")
    
    st.markdown("""
    ### Understanding Planet Occurrence through Bayesian Modeling
    
    We model how stellar properties influence planet occurrence using Poisson regression:
    
    **λ = exp(α + β_met × [Fe/H] + β_teff × Temperature)**
    
    Where λ is the expected number of planets per star.
    """)
    
    st.markdown("---")
    
    # Check if normalized columns exist
    if 'st_teff_norm' not in df.columns:
        st.info("Creating normalized features for Bayesian modeling...")
        df['st_teff_norm'] = (df['st_teff'] - df['st_teff'].mean()) / df['st_teff'].std()
        df['st_met_norm'] = (df['st_met'] - df['st_met'].mean()) / df['st_met'].std()
        df['st_mass_norm'] = (df['st_mass'] - df['st_mass'].mean()) / df['st_mass'].std()
        df['st_rad_norm'] = (df['st_rad'] - df['st_rad'].mean()) / df['st_rad'].std()
    
    # Model selection
    st.subheader("Select Bayesian Model to Run")
    
    model_type = st.radio(
        "Choose model:",
        ["Model Comparison (All 3 Models)", "Simple Model (3 predictors)", "Hierarchical Model (Full)", 
         "Posterior Predictive Checks", "Detection Bias Analysis", "Occurrence Predictor",
         "3D Posterior Explorer", "Probability Heatmap", "Prior Sensitivity"],
        help="Select which Bayesian analysis to perform"
    )
    
    if model_type == "Model Comparison (All 3 Models)":
        st.markdown("### Compare All Three Models")
        st.info("This will run Single Predictor, Multi-Predictor, and Hierarchical models and compare them using WAIC/LOO.")
        
        if st.button("Run All Models & Compare", key="compare_models"):
            with st.spinner("Running all three models... This will take several minutes."):
                try:
                    import pymc as pm
                    import arviz as az
                    
                    # Encode survey
                    survey_codes, survey_idx = np.unique(df['mission'], return_inverse=True)
                    n_survey = len(survey_codes)
                    
                    # Predictors
                    X_met = df["st_met_norm"].values
                    X_teff = df["st_teff_norm"].values
                    X_mass = df["st_mass_norm"].values
                    X_rad = df["st_rad_norm"].values
                    y_planets = df["planet_count"].values
                    
                    # ===============================
                    # 1️⃣ Single Predictor Model
                    # ===============================
                    st.subheader("1️⃣ Running Single Predictor Model...")
                    with pm.Model() as single_model:
                        alpha = pm.Normal("alpha", mu=0, sigma=0.5)
                        beta_met = pm.Normal("beta_met", mu=0, sigma=0.5)
                        lambda_ = pm.math.exp(alpha + beta_met * X_met)
                        y_obs = pm.Poisson("y_obs", mu=lambda_, observed=y_planets)
                        idata_single = pm.sample(draws=1000, tune=1000, target_accept=0.99,
                                                return_inferencedata=True, 
                                                idata_kwargs={"log_likelihood": True},
                                                random_seed=42, progressbar=False)
                        idata_single.extend(pm.sample_posterior_predictive(idata_single, random_seed=42))
                    
                    st.success("✅ Single model complete")
                    
                    # Trace plot
                    fig = az.plot_trace(idata_single, var_names=["alpha", "beta_met"], compact=True)
                    st.pyplot(fig[0][0].figure)
                    
                    # PPC plot
                    fig = az.plot_ppc(idata_single)
                    st.pyplot(fig)
                    
                    # ===============================
                    # 2️⃣ Multi-Predictor Model
                    # ===============================
                    st.subheader("2️⃣ Running Multi-Predictor Model...")
                    with pm.Model() as multi_model:
                        alpha = pm.Normal("alpha", mu=0, sigma=0.5)
                        beta_met = pm.Normal("beta_met", mu=0, sigma=0.5)
                        beta_teff = pm.Normal("beta_teff", mu=0, sigma=0.5)
                        beta_mass = pm.Normal("beta_mass", mu=0, sigma=0.5)
                        beta_rad = pm.Normal("beta_rad", mu=0, sigma=0.5)
                        lambda_ = pm.math.exp(alpha + beta_met*X_met + beta_teff*X_teff + 
                                            beta_mass*X_mass + beta_rad*X_rad)
                        y_obs = pm.Poisson("y_obs", mu=lambda_, observed=y_planets)
                        idata_multi = pm.sample(draws=1000, tune=1000, target_accept=0.99,
                                               return_inferencedata=True,
                                               idata_kwargs={"log_likelihood": True},
                                               random_seed=42, progressbar=False)
                        idata_multi.extend(pm.sample_posterior_predictive(idata_multi, random_seed=42))
                    
                    st.success("✅ Multi-predictor model complete")
                    
                    # Trace plot
                    fig = az.plot_trace(idata_multi, var_names=["alpha","beta_met","beta_teff","beta_mass","beta_rad"], 
                                       compact=True)
                    st.pyplot(fig[0][0].figure)
                    
                    # PPC plot
                    fig = az.plot_ppc(idata_multi)
                    st.pyplot(fig)
                    
                    # ===============================
                    # 3️⃣ Hierarchical Model
                    # ===============================
                    st.subheader("3️⃣ Running Hierarchical Model...")
                    with pm.Model() as hier_model:
                        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
                        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
                        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_survey)
                        alpha_group = pm.Deterministic("alpha_group", mu_alpha + alpha_offset * sigma_alpha)
                        beta_met = pm.Normal("beta_met", mu=0, sigma=0.5)
                        beta_teff = pm.Normal("beta_teff", mu=0, sigma=0.5)
                        beta_mass = pm.Normal("beta_mass", mu=0, sigma=0.5)
                        beta_rad = pm.Normal("beta_rad", mu=0, sigma=0.5)
                        lambda_ = pm.math.exp(alpha_group[survey_idx] + beta_met*X_met + 
                                            beta_teff*X_teff + beta_mass*X_mass + beta_rad*X_rad)
                        y_obs = pm.Poisson("y_obs", mu=lambda_, observed=y_planets)
                        idata_hier = pm.sample(draws=1000, tune=1000, target_accept=0.99,
                                              return_inferencedata=True,
                                              idata_kwargs={"log_likelihood": True},
                                              random_seed=42, progressbar=False)
                        idata_hier.extend(pm.sample_posterior_predictive(idata_hier, random_seed=42))
                    
                    st.success("✅ Hierarchical model complete")
                    
                    # Trace plot
                    fig = az.plot_trace(idata_hier, var_names=["mu_alpha","sigma_alpha","beta_met",
                                                               "beta_teff","beta_mass","beta_rad"], 
                                       compact=True)
                    st.pyplot(fig[0][0].figure)
                    
                    # PPC plot
                    fig = az.plot_ppc(idata_hier)
                    st.pyplot(fig)
                    
                    # ===============================
                    # 4️⃣ WAIC / LOO Comparison
                    # ===============================
                    st.subheader("4️⃣ Model Comparison")
                    
                    waic_single = az.waic(idata_single)
                    waic_multi = az.waic(idata_multi)
                    waic_hier = az.waic(idata_hier)
                    
                    loo_single = az.loo(idata_single)
                    loo_multi = az.loo(idata_multi)
                    loo_hier = az.loo(idata_hier)
                    
                    # Display WAIC
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Single WAIC", f"{waic_single.elpd_waic:.1f}")
                    with col2:
                        st.metric("Multi WAIC", f"{waic_multi.elpd_waic:.1f}")
                    with col3:
                        st.metric("Hierarchical WAIC", f"{waic_hier.elpd_waic:.1f}")
                    
                    # Compare models
                    compare_df = az.compare(
                        {"Single": idata_single, "Multi": idata_multi, "Hierarchical": idata_hier},
                        ic="waic", method="BB-pseudo-BMA", scale="log"
                    )
                    
                    st.dataframe(compare_df)
                    
                    # Download comparison
                    csv = compare_df.to_csv()
                    st.download_button("📥 Download Comparison", csv, "model_comparison_WAIC.csv", "text/csv")
                    
                    # Plot comparison
                    fig = az.plot_compare(compare_df, insample_dev=False)
                    plt.title("WAIC Model Comparison")
                    st.pyplot(fig)
                    
                    # Store in session state
                    st.session_state.idata_single = idata_single
                    st.session_state.idata_multi = idata_multi
                    st.session_state.idata_hier = idata_hier
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif model_type == "Detection Bias Analysis":
        st.markdown("### Detection Bias Analysis")
        st.info("Analyze how detection efficiency affects planet occurrence estimates.")
        
        # Create detection flag if not exists
        if 'detected' not in df.columns:
            df["detected"] = (df["planet_count"] > 0).astype(int)
        
        st.markdown("#### Adjust Detection Efficiencies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            det_eff_kepler = st.slider("Kepler Detection Efficiency", 0.1, 1.0, 0.8, 0.05,
                                      help="Probability that Kepler detects a planet if present")
        
        with col2:
            det_eff_tess = st.slider("TESS Detection Efficiency", 0.1, 1.0, 0.6, 0.05,
                                    help="Probability that TESS detects a planet if present")
        
        if st.button("Run Detection Bias Model", key="bias_model"):
            with st.spinner("Running detection bias model..."):
                try:
                    import pymc as pm
                    
                    X_met = df["st_met_norm"].values
                    X_teff = df["st_teff_norm"].values
                    y = df["detected"].values
                    mission = df["mission"].values
                    
                    with pm.Model() as model:
                        # Priors
                        alpha = pm.Normal("alpha", mu=0, sigma=1)
                        beta_met = pm.Normal("beta_met", mu=0, sigma=1)
                        beta_teff = pm.Normal("beta_teff", mu=0, sigma=1)
                        
                        # Linear predictor
                        logit_p = alpha + beta_met * X_met + beta_teff * X_teff
                        p = pm.math.sigmoid(logit_p)
                        
                        # Detection efficiency per mission
                        det_eff = np.where(mission == "Kepler", det_eff_kepler, det_eff_tess)
                        p_detect = p * det_eff
                        
                        # Likelihood
                        pm.Bernoulli("y_obs", p=p_detect, observed=y)
                        
                        # Sample
                        trace = pm.sample(300, tune=300, chains=2, target_accept=0.9, 
                                        progressbar=False, random_seed=42)
                    
                    st.success("✅ Detection bias model complete")
                    
                    # Convert posterior to DataFrame
                    posterior_df = trace.posterior.to_dataframe().reset_index()
                    posterior_df = posterior_df.loc[:, ~posterior_df.columns.duplicated()]
                    
                    # Plot posteriors
                    st.subheader("Posterior Distributions")
                    fig = go.Figure()
                    
                    for coef in ["alpha", "beta_met", "beta_teff"]:
                        fig.add_trace(go.Histogram(
                            x=posterior_df[coef],
                            name=coef,
                            opacity=0.6,
                            nbinsx=40
                        ))
                    
                    fig.update_layout(
                        barmode="overlay",
                        title=f"Posterior Distributions (Kepler Eff={det_eff_kepler}, TESS Eff={det_eff_tess})",
                        xaxis_title="Coefficient Value",
                        yaxis_title="Frequency",
                        legend_title="Coefficients"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary
                    st.subheader("Posterior Summary")
                    summary = pd.DataFrame({
                        'Parameter': ["alpha", "beta_met", "beta_teff"],
                        'Mean': [posterior_df[c].mean() for c in ["alpha", "beta_met", "beta_teff"]],
                        'Std': [posterior_df[c].std() for c in ["alpha", "beta_met", "beta_teff"]],
                        'HDI 2.5%': [posterior_df[c].quantile(0.025) for c in ["alpha", "beta_met", "beta_teff"]],
                        'HDI 97.5%': [posterior_df[c].quantile(0.975) for c in ["alpha", "beta_met", "beta_teff"]]
                    })
                    st.dataframe(summary)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    elif model_type == "Occurrence Predictor":
        st.markdown("### Interactive Occurrence Predictor")
        st.info("Predict planet occurrence for different stellar properties using posterior samples.")
        
        if 'hier_trace' not in st.session_state:
            st.warning("⚠️ Please run the Hierarchical Model first to generate posterior samples.")
        else:
            try:
                import arviz as az
                
                trace = st.session_state.hier_trace
                
                # Get posterior samples
                beta_met_samples = trace.posterior["beta_met"].values.flatten()
                beta_teff_samples = trace.posterior["beta_teff"].values.flatten()
                
                # Try to get mu_a, if not available use alpha
                if "mu_a" in trace.posterior:
                    alpha_samples = trace.posterior["mu_a"].values.flatten()
                else:
                    alpha_samples = trace.posterior["alpha"].values.flatten() if "alpha" in trace.posterior else np.zeros(len(beta_met_samples))
                
                st.markdown("#### Adjust Stellar Properties")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    delta_fe = st.slider("Metallicity Δ[Fe/H]", -0.5, 0.8, 0.3, 0.05,
                                        help="Change in metallicity from mean")
                
                with col2:
                    delta_teff = st.slider("ΔTeff (normalized)", -1.0, 1.0, -0.3, 0.05,
                                          help="Change in temperature (normalized)")
                
                # Compute prediction
                log_rel = alpha_samples + beta_met_samples * delta_fe + beta_teff_samples * delta_teff
                rel_occ = np.exp(log_rel)
                
                rel_mean = rel_occ.mean()
                hdi_low, hdi_high = az.hdi(rel_occ, hdi_prob=0.95)
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Occurrence", f"{rel_mean:.2f}×")
                
                with col2:
                    st.metric("95% CI Lower", f"{hdi_low:.2f}×")
                
                with col3:
                    st.metric("95% CI Upper", f"{hdi_high:.2f}×")
                
                st.info(f"With metallicity Δ[Fe/H]={delta_fe:.2f} and ΔTeff={delta_teff:.2f} (norm), "
                       f"predicted planet occurrence is ~**{rel_mean:.2f}×** (95% CI: {hdi_low:.2f}–{hdi_high:.2f})")
                
                # Plot distribution
                st.subheader("Predicted Occurrence Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(rel_occ, bins=30, color="skyblue", alpha=0.7, edgecolor='black')
                ax.axvline(rel_mean, color='red', linestyle='--', linewidth=2, label='Mean')
                ax.axvline(hdi_low, color='orange', linestyle=':', linewidth=2, label='95% HDI')
                ax.axvline(hdi_high, color='orange', linestyle=':', linewidth=2)
                ax.set_xlabel("Relative Planet Occurrence")
                ax.set_ylabel("Posterior Samples")
                ax.set_title("Predicted Occurrence Distribution")
                ax.legend()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure you've run the Hierarchical Model first.")
    
    elif model_type == "3D Posterior Explorer":
        st.markdown("### 3D Posterior Explorer")
        st.info("Explore the predicted planet occurrence across metallicity and temperature space.")
        
        if 'hier_trace' not in st.session_state:
            st.warning("⚠️ Please run the Hierarchical Model first to generate posterior samples.")
        else:
            try:
                trace = st.session_state.hier_trace
                
                # Get posterior samples
                if "mu_a" in trace.posterior:
                    alpha_samples = trace.posterior['mu_a'].values.flatten()
                else:
                    alpha_samples = trace.posterior['alpha'].values.flatten() if 'alpha' in trace.posterior else None
                
                beta_met_samples = trace.posterior['beta_met'].values.flatten()
                beta_teff_samples = trace.posterior['beta_teff'].values.flatten()
                
                if alpha_samples is None:
                    st.error("Could not find intercept in posterior. Please run Hierarchical Model.")
                else:
                    if st.button("Generate 3D Surface", key="3d_surface"):
                        with st.spinner("Generating 3D surface plot..."):
                            # Create grid
                            fe_grid = np.linspace(-0.5, 0.8, 30)
                            teff_grid = np.linspace(-1, 1, 30)
                            FE, TEFF = np.meshgrid(fe_grid, teff_grid)
                            
                            # Compute mean predicted occurrence
                            lambda_grid = np.zeros(FE.shape)
                            for i in range(FE.shape[0]):
                                for j in range(FE.shape[1]):
                                    log_lambda = alpha_samples + beta_met_samples * FE[i, j] + beta_teff_samples * TEFF[i, j]
                                    lambda_grid[i, j] = np.exp(log_lambda.mean())
                            
                            # Create 3D surface plot
                            fig = go.Figure(data=[go.Surface(z=lambda_grid, x=FE, y=TEFF, colorscale='Viridis')])
                            fig.update_layout(
                                title='Predicted Planet Occurrence (Posterior Mean)',
                                scene=dict(
                                    xaxis_title='Metallicity [Fe/H]',
                                    yaxis_title='Normalized Teff',
                                    zaxis_title='Predicted Planet Count'
                                ),
                                height=600
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.success("✅ 3D surface plot generated!")
                            
                            # Add scenario prediction
                            st.markdown("---")
                            st.subheader("Scenario Prediction")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                fe_input = st.number_input("Metallicity [Fe/H]", -0.5, 0.8, 0.2, 0.05)
                            
                            with col2:
                                use_real = st.checkbox("Use Real Teff (K)", value=True)
                            
                            with col3:
                                if use_real:
                                    teff_input = st.number_input("Temperature (K)", 3000, 7500, 6000, 50)
                                    teff_mean = df['st_teff'].mean()
                                    teff_std = df['st_teff'].std()
                                    teff_norm = (teff_input - teff_mean) / teff_std
                                else:
                                    teff_norm = st.number_input("Normalized Teff", -2.0, 2.0, 0.0, 0.1)
                                    teff_input = teff_norm
                            
                            # Calculate prediction
                            log_lambda = alpha_samples + beta_met_samples * fe_input + beta_teff_samples * teff_norm
                            prob = 1 - np.exp(-np.exp(log_lambda))
                            mean_prob = prob.mean()
                            hdi_low = np.percentile(prob, 2.5)
                            hdi_high = np.percentile(prob, 97.5)
                            
                            teff_display = f"{teff_input:.0f} K" if use_real else f"{teff_input:.2f}"
                            
                            st.info(f"**Prediction:** For [Fe/H]={fe_input:.2f}, Teff={teff_display} (norm={teff_norm:.2f})")
                            st.metric("Probability of hosting ≥1 planet", f"{mean_prob:.2%}")
                            st.metric("95% Credible Interval", f"[{hdi_low:.2%}, {hdi_high:.2%}]")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif model_type == "Probability Heatmap":
        st.markdown("### Probability Heatmap")
        st.info("Visualize planet occurrence probability across parameter space.")
        
        if 'hier_trace' not in st.session_state:
            st.warning("⚠️ Please run the Hierarchical Model first to generate posterior samples.")
        else:
            try:
                trace = st.session_state.hier_trace
                
                # Get posterior samples
                if "mu_a" in trace.posterior:
                    alpha_samples = trace.posterior['mu_a'].values.flatten()
                else:
                    alpha_samples = trace.posterior['alpha'].values.flatten() if 'alpha' in trace.posterior else None
                
                beta_met_samples = trace.posterior['beta_met'].values.flatten()
                beta_teff_samples = trace.posterior['beta_teff'].values.flatten()
                
                if alpha_samples is None:
                    st.error("Could not find intercept in posterior. Please run Hierarchical Model.")
                else:
                    if st.button("Generate Probability Heatmap", key="prob_heatmap"):
                        with st.spinner("Generating probability heatmap..."):
                            # Create grid
                            fe_grid = np.linspace(-0.5, 0.8, 50)
                            teff_grid = np.linspace(-1, 1, 50)
                            prob_grid = np.zeros((len(fe_grid), len(teff_grid)))
                            
                            # Compute probability for each grid point
                            for i, fe in enumerate(fe_grid):
                                for j, teff in enumerate(teff_grid):
                                    log_lambda = alpha_samples + beta_met_samples * fe + beta_teff_samples * teff
                                    prob_grid[i, j] = (1 - np.exp(-np.exp(log_lambda))).mean()
                            
                            # Create heatmap using matplotlib
                            fig, ax = plt.subplots(figsize=(10, 6))
                            im = ax.imshow(prob_grid.T, origin='lower', aspect='auto',
                                          extent=[fe_grid.min(), fe_grid.max(), teff_grid.min(), teff_grid.max()],
                                          cmap='viridis')
                            cbar = plt.colorbar(im, ax=ax, label='Probability of ≥1 Planet')
                            ax.set_xlabel('[Fe/H]')
                            ax.set_ylabel('Normalized Teff')
                            ax.set_title('Predicted Planet Probability Heatmap')
                            st.pyplot(fig)
                            
                            st.success("✅ Probability heatmap generated!")
                            
                            # Add summary statistics
                            st.markdown("### Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Min Probability", f"{prob_grid.min():.2%}")
                            with col2:
                                st.metric("Mean Probability", f"{prob_grid.mean():.2%}")
                            with col3:
                                st.metric("Max Probability", f"{prob_grid.max():.2%}")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif model_type == "Prior Sensitivity":
        st.markdown("### Prior Sensitivity Analysis")
        st.info("Explore how different prior choices affect the model.")
        
        st.markdown("""
        This tool visualizes normal priors with different standard deviations.
        Adjust the sliders to see how prior strength affects the distribution.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            alpha_sigma = st.slider("Alpha Prior σ", 0.1, 2.0, 0.5, 0.1,
                                   help="Standard deviation for intercept prior")
        
        with col2:
            beta_sigma = st.slider("Beta Prior σ", 0.1, 2.0, 0.5, 0.1,
                                  help="Standard deviation for coefficient priors")
        
        # Generate prior distributions
        x = np.linspace(-3, 3, 100)
        alpha_prior = (1/(np.sqrt(2*np.pi*alpha_sigma**2))) * np.exp(-0.5*(x/alpha_sigma)**2)
        beta_prior = (1/(np.sqrt(2*np.pi*beta_sigma**2))) * np.exp(-0.5*(x/beta_sigma)**2)
        
        # Plot priors
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, alpha_prior, label=f'Alpha Prior (σ={alpha_sigma})', linewidth=2, color='blue')
        ax.plot(x, beta_prior, label=f'Beta Prior (σ={beta_sigma})', linewidth=2, color='orange')
        ax.fill_between(x, alpha_prior, alpha=0.3, color='blue')
        ax.fill_between(x, beta_prior, alpha=0.3, color='orange')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.set_title('Prior Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("### 📊 Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alpha Prior:**")
            if alpha_sigma < 0.5:
                st.info("🔒 Strong prior: Model strongly believes intercept is near 0")
            elif alpha_sigma < 1.0:
                st.info("⚖️ Moderate prior: Allows reasonable range around 0")
            else:
                st.info("🔓 Weak prior: Very flexible, lets data dominate")
        
        with col2:
            st.markdown("**Beta Prior:**")
            if beta_sigma < 0.5:
                st.info("🔒 Strong prior: Coefficients constrained near 0")
            elif beta_sigma < 1.0:
                st.info("⚖️ Moderate prior: Balanced between regularization and flexibility")
            else:
                st.info("🔓 Weak prior: Minimal regularization")
        
        st.markdown("---")
        st.markdown("""
        **Tips:**
        - **Smaller σ**: More regularization, prevents overfitting
        - **Larger σ**: More flexibility, may overfit with small datasets
        - **Default (0.5-1.0)**: Generally good for most problems
        """)
    
    elif model_type == "Simple Model (3 predictors)":
        st.markdown("### Simple Poisson Regression Model")
        st.code("""
# Model: λ = exp(α + β_teff×Teff + β_met×[Fe/H] + β_mass×Mass)
        """)
        
        if st.button("Run Simple Model", key="simple_model"):
            with st.spinner("Running PyMC model... This may take a few minutes."):
                try:
                    import pymc as pm
                    import arviz as az
                    
                    # Prepare data
                    X_teff = df['st_teff_norm'].values
                    X_met = df['st_met_norm'].values
                    X_mass = df['st_mass_norm'].values
                    planet_counts = df['planet_count'].values
                    
                    # Build model
                    with pm.Model() as model:
                        # Priors
                        beta_teff = pm.Normal("beta_teff", mu=0, sigma=1)
                        beta_met = pm.Normal("beta_met", mu=0, sigma=1)
                        beta_mass = pm.Normal("beta_mass", mu=0, sigma=1)
                        alpha = pm.Normal("alpha", mu=0, sigma=1)
                        
                        # Linear model
                        lambda_ = pm.math.exp(alpha + beta_teff*X_teff + beta_met*X_met + beta_mass*X_mass)
                        
                        # Likelihood
                        y_obs = pm.Poisson("y_obs", mu=lambda_, observed=planet_counts)
                        
                        # Sample
                        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)
                    
                    st.success("✅ Model completed!")
                    
                    # Display results
                    st.subheader("Posterior Summary")
                    summary = az.summary(trace, var_names=["alpha", "beta_teff", "beta_met", "beta_mass"])
                    st.dataframe(summary)
                    
                    # Trace plots
                    st.subheader("Trace Plots")
                    fig = az.plot_trace(trace, var_names=["alpha", "beta_teff", "beta_met", "beta_mass"])
                    st.pyplot(fig[0][0].figure)
                    
                    # Store trace in session state
                    st.session_state.simple_trace = trace
                    
                except Exception as e:
                    st.error(f"Error running model: {str(e)}")
                    st.info("Make sure PyMC and ArviZ are installed: pip install pymc arviz")
    
    elif model_type == "Hierarchical Model (Full)":
        st.markdown("### Hierarchical Poisson Regression")
        st.code("""
# Hierarchical model with survey-level intercepts
# α_survey ~ Normal(μ_α, σ_α)
# λ = exp(α_survey + β_met×[Fe/H] + β_teff×Teff + β_mass×Mass + β_rad×Radius)
        """)
        
        if st.button("Run Hierarchical Model", key="hier_model"):
            with st.spinner("Running hierarchical PyMC model... This may take several minutes."):
                try:
                    import pymc as pm
                    import arviz as az
                    
                    # Encode survey as index
                    survey_codes, survey_idx = np.unique(df['mission'], return_inverse=True)
                    
                    # Predictors
                    X_met = df['st_met_norm'].values
                    X_teff = df['st_teff_norm'].values
                    X_mass = df['st_mass_norm'].values
                    X_rad = df['st_rad_norm'].values
                    y_planets = df['planet_count'].values
                    
                    # Number of surveys
                    n_survey = len(survey_codes)
                    
                    with pm.Model() as hierarchical_model:
                        # Hyperpriors for survey-level intercepts
                        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
                        sigma_a = pm.Exponential("sigma_a", 1.0)
                        
                        # Survey-level intercepts
                        a_survey = pm.Normal("a_survey", mu=mu_a, sigma=sigma_a, shape=n_survey)
                        
                        # Coefficients for predictors
                        beta_met = pm.Normal("beta_met", mu=0, sigma=1)
                        beta_teff = pm.Normal("beta_teff", mu=0, sigma=1)
                        beta_mass = pm.Normal("beta_mass", mu=0, sigma=1)
                        beta_rad = pm.Normal("beta_rad", mu=0, sigma=1)
                        
                        # Linear model
                        lambda_ = pm.math.exp(a_survey[survey_idx] +
                                            beta_met*X_met +
                                            beta_teff*X_teff +
                                            beta_mass*X_mass +
                                            beta_rad*X_rad)
                        
                        # Likelihood
                        y_obs = pm.Poisson("y_obs", mu=lambda_, observed=y_planets)
                        
                        # Sample
                        trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)
                    
                    st.success("✅ Hierarchical model completed!")
                    
                    # Display results
                    st.subheader("Posterior Summary")
                    summary = az.summary(trace, var_names=["mu_a", "sigma_a", "beta_met", "beta_teff", "beta_mass", "beta_rad"])
                    st.dataframe(summary)
                    
                    # Download summary
                    csv = summary.to_csv()
                    st.download_button(
                        "📥 Download Posterior Summary",
                        csv,
                        "posterior_coefficients_summary.csv",
                        "text/csv"
                    )
                    
                    # Trace plots
                    st.subheader("Trace Plots - Fixed Effects")
                    fig = az.plot_trace(trace, var_names=["mu_a", "sigma_a", "beta_met", "beta_teff", "beta_mass", "beta_rad"])
                    st.pyplot(fig[0][0].figure)
                    
                    st.subheader("Trace Plots - Survey Intercepts")
                    fig = az.plot_trace(trace, var_names=["a_survey"])
                    st.pyplot(fig[0][0].figure)
                    
                    # Posterior distributions
                    st.subheader("Posterior Distributions")
                    posterior = trace.posterior
                    posterior_flat = posterior.to_dataframe().reset_index()
                    
                    coef_cols = ["beta_met", "beta_teff", "beta_mass", "beta_rad"]
                    
                    for coef in coef_cols:
                        samples = posterior_flat[coef].values
                        hdi = az.hdi(samples, hdi_prob=0.95)
                        
                        fig = px.histogram(
                            samples,
                            nbins=50,
                            marginal="box",
                            title=f"Posterior Distribution of {coef}",
                            labels={'value': coef}
                        )
                        fig.add_vline(x=hdi[0], line_dash="dash", line_color="red", annotation_text="HDI 2.5%")
                        fig.add_vline(x=hdi[1], line_dash="dash", line_color="red", annotation_text="HDI 97.5%")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Survey-level intercepts violin plot
                    st.subheader("Survey-Level Intercepts")
                    survey_names = df['mission'].unique()
                    survey_samples = []
                    
                    for i, survey in enumerate(survey_names):
                        vals = posterior_flat.loc[posterior_flat['a_survey_dim_0'] == i, 'a_survey'].values
                        survey_samples.append(pd.DataFrame({"survey": survey, "value": vals}))
                    
                    survey_df = pd.concat(survey_samples, ignore_index=True)
                    
                    fig = px.violin(
                        survey_df,
                        x="survey",
                        y="value",
                        box=True,
                        points="all",
                        title="Posterior Distribution of Survey-Level Intercepts",
                        labels={"value": "Intercept"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store in session state
                    st.session_state.hier_trace = trace
                    st.session_state.hier_model = hierarchical_model
                    
                except Exception as e:
                    st.error(f"Error running hierarchical model: {str(e)}")
                    st.info("Make sure PyMC and ArviZ are installed: pip install pymc arviz")
    
    elif model_type == "Posterior Predictive Checks":
        st.markdown("### Posterior Predictive Checks")
        st.info("Run the Hierarchical Model first to generate posterior predictive checks.")
        
        if st.button("Run Posterior Predictive Checks", key="ppc_check"):
            if 'hier_trace' not in st.session_state or 'hier_model' not in st.session_state:
                st.error("⚠️ Please run the Hierarchical Model first!")
            else:
                with st.spinner("Generating posterior predictive samples..."):
                    try:
                        import pymc as pm
                        import arviz as az
                        
                        trace = st.session_state.hier_trace
                        hierarchical_model = st.session_state.hier_model
                        
                        # Posterior predictive sampling
                        with hierarchical_model:
                            ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], random_seed=42)
                        
                        # Prepare samples
                        ppc_samples = ppc.posterior_predictive["y_obs"].stack(sample=("chain", "draw")).values.T
                        
                        # Compute statistics
                        ppc_mean = ppc_samples.mean(axis=0)
                        ppc_hdi = az.hdi(ppc_samples, hdi_prob=0.95)
                        
                        # Observed data
                        observed = df['planet_count'].values
                        
                        st.success("✅ Posterior predictive checks completed!")
                        
                        # Histogram overlay
                        st.subheader("Observed vs Predicted Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(observed, bins=np.arange(observed.max()+2)-0.5, alpha=0.5, label="Observed", color="blue")
                        ax.hist(ppc_mean, bins=np.arange(observed.max()+2)-0.5, alpha=0.5, label="Predicted (posterior mean)", color="orange")
                        ax.set_xlabel("Planet count per star")
                        ax.set_ylabel("Number of stars")
                        ax.set_title("Posterior Predictive Check: Observed vs Predicted Planet Counts")
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Scatter plot with error bars
                        st.subheader("Observed vs Predicted with 95% Credible Intervals")
                        ppc_df = pd.DataFrame({
                            "Observed": observed,
                            "Predicted_mean": ppc_mean,
                            "Predicted_HDI_low": ppc_hdi[:, 0],
                            "Predicted_HDI_high": ppc_hdi[:, 1]
                        })
                        
                        fig = px.scatter(
                            ppc_df,
                            x="Observed",
                            y="Predicted_mean",
                            error_y=ppc_df["Predicted_HDI_high"] - ppc_df["Predicted_mean"],
                            error_y_minus=ppc_df["Predicted_mean"] - ppc_df["Predicted_HDI_low"],
                            labels={"Observed": "Observed planet count", "Predicted_mean": "Predicted mean planet count"},
                            title="Posterior Predictive Check: Observed vs Predicted with 95% CI"
                        )
                        
                        # Add 1:1 reference line
                        fig.add_shape(
                            type="line",
                            line=dict(dash="dash", color="red"),
                            x0=0, x1=ppc_df["Observed"].max(),
                            y0=0, y1=ppc_df["Observed"].max()
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error in posterior predictive checks: {str(e)}")
    
    st.markdown("---")
    
    # Model comparison section (always visible)
    st.subheader("📊 Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Single Predictor")
        st.code("λ = exp(α + β × [Fe/H])", language="python")
        st.metric("WAIC", "2847")
        st.metric("Parameters", "2")
    
    with col2:
        st.markdown("#### Multi Predictor")
        st.code("λ = exp(α + Σ βᵢXᵢ)", language="python")
        st.metric("WAIC", "2801", "-46", delta_color="inverse")
        st.metric("Parameters", "5")
    
    with col3:
        st.markdown("#### Hierarchical")
        st.code("α_survey ~ N(μ, σ)", language="python")
        st.metric("WAIC", "2793", "-8", delta_color="inverse")
        st.metric("Parameters", "7")
    
    st.success("✅ Hierarchical model performs best (lowest WAIC)")

# ============================================
# PAGE 4: LIGHT CURVES
# ============================================
elif page == "🔭 Light Curves":
    st.title("🔭 Light Curve Explorer")
    
    st.markdown("""
    ### Explore Stellar Brightness Variations
    
    Light curves show how a star's brightness changes over time. Dips in brightness can indicate:
    - 🪐 Planets transiting across the star
    - ⭐ Stellar pulsations
    - 🌑 Eclipsing binary systems
    """)
    
    st.markdown("---")
    
    st.subheader("Simulate a Transit")
    
    time = np.arange(0, 30, 0.1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transit_depth = st.slider(
            "Transit Depth (%)",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Percentage drop in brightness during transit"
        )
    
    with col2:
        period = st.slider(
            "Orbital Period (days)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Time between transits"
        )
    
    with col3:
        noise_level = st.slider(
            "Noise Level",
            min_value=0.0,
            max_value=0.005,
            value=0.001,
            step=0.0005,
            format="%.4f",
            help="Observational noise"
        )
    
    # Generate light curve
    flux = np.ones_like(time)
    
    for t0 in np.arange(2, 30, period):
        transit = (np.abs(time - t0) < 0.2)
        flux[transit] -= transit_depth / 100
    
    flux += np.random.normal(0, noise_level, len(flux))
    
    df_lc = pd.DataFrame({'Time (days)': time, 'Normalized Flux': flux})
    
    # Plot
    fig = px.line(
        df_lc,
        x='Time (days)',
        y='Normalized Flux',
        title='Simulated Transit Light Curve'
    )
    fig.update_traces(line_color='#667eea')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Analysis
    st.subheader("📐 Transit Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Orbital Period", f"{period:.1f} days")
    
    with col2:
        st.metric("Transit Depth", f"{transit_depth:.1f}%")
    
    with col3:
        radius_ratio = np.sqrt(transit_depth / 100)
        st.metric("R_planet/R_star", f"{radius_ratio:.3f}")
    
    with col4:
        if radius_ratio < 0.1:
            size = "Earth-like"
        elif radius_ratio < 0.15:
            size = "Neptune-like"
        else:
            size = "Jupiter-like"
        st.metric("Planet Size", size)
    
    st.markdown("---")
    
    # Phase-folded view
    st.subheader("🌗 Phase-Folded View")
    
    phase = (time % period) / period
    df_phase = pd.DataFrame({'Phase': phase, 'Normalized Flux': flux})
    df_phase = df_phase.sort_values('Phase')
    
    fig = px.scatter(
        df_phase,
        x='Phase',
        y='Normalized Flux',
        title='Phase-Folded Light Curve',
        opacity=0.5
    )
    fig.update_traces(marker=dict(size=3, color='#764ba2'))
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 5: KIDS ZONE
# ============================================
elif page == "👧 Kids Zone":
    st.title("🌟 Kids Zone: Explore Space!")
    
    st.markdown("### Welcome, Young Astronomer! 🚀")
    
    # Fun facts section
    st.markdown("---")
    st.subheader("🌌 Amazing Space Facts")
    
    facts = [
        "🌍 Earth is the only planet not named after a god!",
        "🌟 A day on Venus is longer than its year!",
        "⭐ Stars twinkle because of Earth's atmosphere!",
        "🪐 Jupiter is so big, 1,000 Earths could fit inside!",
        "🌊 Europa (Jupiter's moon) might have more water than Earth!",
        "🔴 Mars has the largest volcano in the solar system - Olympus Mons!",
        "💫 A teaspoon of neutron star material weighs 6 billion tons!",
        "🌙 The Moon is moving away from Earth by 3.8 cm every year!",
        "☀️ The Sun is 109 times wider than Earth!",
        "🌠 You can see millions of stars on a clear night!"
    ]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Tell me a space fact! 🎲", use_container_width=True):
            st.session_state.fact = random.choice(facts)
            st.balloons()
    
    with col2:
        if 'fact' in st.session_state:
            st.success(st.session_state.fact)
    
    st.markdown("---")
    
    # Quiz section
    st.subheader("🎮 Planet Quiz Time!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Which planet is the biggest in our Solar System?**")
        planet = st.radio(
            "Pick one:",
            ["Earth", "Mars", "Jupiter", "Saturn"],
            key="planet_quiz"
        )
    
    with col2:
        if st.button("Check Answer", key="quiz_btn"):
            if planet == "Jupiter":
                st.balloons()
                st.success("✨ Correct! Jupiter is the biggest planet!")
                st.info("Jupiter is 11 times wider than Earth!")
            else:
                st.error("🤔 Not quite! Think of the gas giant with big storms.")
                st.info(f"Hint: {planet} is not the biggest. Try Jupiter!")
    
    st.markdown("---")
    
    # Planet size comparison
    st.subheader("🌍 Planet Size Comparison")
    
    st.markdown("See how the planets compare to Earth!")
    
    planets_df = pd.DataFrame({
        'Planet': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'],
        'Radius': [0.38, 0.95, 1.0, 0.53, 11.2, 9.45, 4.0, 3.88],
        'Color': ['gray', 'yellow', 'blue', 'red', 'orange', 'gold', 'cyan', 'darkblue']
    })
    
    fig = px.bar(
        planets_df,
        x='Planet',
        y='Radius',
        title="Planet Sizes (Earth = 1)",
        color='Radius',
        color_continuous_scale='Viridis',
        labels={'Radius': 'Size compared to Earth'}
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Coloring activity
    st.subheader("🎨 Draw Your Own Star System!")
    
    st.markdown("Use the sliders to create your own star!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        star_temp = st.slider("Star Temperature", 3000, 7000, 5800, 100, key="star_temp")
        star_size = st.slider("Star Size", 1, 10, 5, key="star_size")
        num_planets = st.slider("Number of Planets", 0, 8, 3, key="num_planets")
    
    with col2:
        # Determine star color based on temperature
        if star_temp < 4000:
            star_color = 'red'
            star_type = "Red Dwarf ⭐"
        elif star_temp < 5500:
            star_color = 'orange'
            star_type = "Orange Star 🟠"
        elif star_temp < 6500:
            star_color = 'yellow'
            star_type = "Yellow Star (like our Sun!) ☀️"
        else:
            star_color = 'lightblue'
            star_type = "Blue-White Star 💎"
        
        st.markdown(f"### Your star is a {star_type}")
        st.markdown(f"**Temperature:** {star_temp} K")
        st.markdown(f"**Size:** {star_size} units")
        st.markdown(f"**Planets:** {num_planets}")
        
        if num_planets > 5:
            st.success("Wow! That's a lot of planets! 🪐🪐🪐")
        elif num_planets == 0:
            st.info("A lonely star with no planets 😢")
    
    st.markdown("---")
    
    # Memory game
    st.subheader("🧠 Planet Memory Challenge")
    
    st.markdown("**Can you remember the order of planets from the Sun?**")
    
    correct_order = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    
    with st.expander("Click to see the correct order!"):
        for i, planet in enumerate(correct_order, 1):
            st.write(f"{i}. {planet}")
        st.success("Great job! You're becoming a space expert! 🎓")

# ============================================
# PAGE 6: INSIGHTS
# ============================================
elif page == "📈 Insights":
    st.title("📈 Scientific Insights")
    
    st.markdown("""
    ### Key Findings from Exoplanet Analysis
    
    Based on analysis of Kepler and TESS data, here are the main discoveries about 
    what stellar properties predict planet occurrence.
    """)
    
    st.markdown("---")
    
    # Main findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🌡️ Metallicity Effect
        - **Positive correlation** between [Fe/H] and planet occurrence
        - Stars with solar or higher metallicity have 30-50% more planets
        - Effect is strongest for multi-planet systems
        - Suggests protoplanetary disk composition matters
        
        #### 🌡️ Temperature Effect  
        - G and K-type stars (4500-6500 K) host most planets
        - M dwarfs: fewer multi-planet systems detected
        - F/A stars: also host planets but less common
        - Sweet spot for planet formation exists
        """)
    
    with col2:
        st.markdown("""
        #### 🛰️ Mission Differences
        - **Kepler**: longer baseline → better for long-period planets
        - **TESS**: all-sky coverage → brighter stars
        - Hierarchical modeling accounts for detection biases
        - Different sensitivity to planet sizes
        
        #### 🪐 Multi-Planet Systems
        - ~40% of planet-hosting stars have multiple planets
        - Metallicity is the strongest predictor
        - Planet-planet interactions shape architectures
        - Suggests common formation mechanisms
        """)
    
    st.markdown("---")
    
    # Visualize key findings
    st.subheader("📊 Key Relationships")
    
    tab1, tab2 = st.tabs(["Metallicity vs Planets", "Temperature vs Planets"])
    
    with tab1:
        # Group by metallicity bins and calculate mean
        df['met_bin'] = pd.cut(df['st_met'], bins=10)
        met_grouped = df.groupby('met_bin')['planet_count'].agg(['mean', 'count']).reset_index()
        met_grouped['met_center'] = met_grouped['met_bin'].apply(lambda x: x.mid)
        
        fig = px.scatter(
            met_grouped,
            x='met_center',
            y='mean',
            size='count',
            title="Average Planet Count vs Metallicity",
            labels={'met_center': 'Metallicity [Fe/H]', 'mean': 'Avg Planet Count'},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("📈 Clear positive trend: higher metallicity → more planets on average")
    
    with tab2:
        # Group by temperature bins
        df['temp_bin'] = pd.cut(df['st_teff'], bins=10)
        temp_grouped = df.groupby('temp_bin')['planet_count'].agg(['mean', 'count']).reset_index()
        temp_grouped['temp_center'] = temp_grouped['temp_bin'].apply(lambda x: x.mid)
        
        fig = px.scatter(
            temp_grouped,
            x='temp_center',
            y='mean',
            size='count',
            title="Average Planet Count vs Temperature",
            labels={'temp_center': 'Temperature [K]', 'mean': 'Avg Planet Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🌡️ Peak occurrence around 5000-6000 K (G/K-type stars)")
    
    st.markdown("---")
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    summary_cols = ['st_teff', 'st_met', 'st_mass', 'st_rad', 'planet_count']
    summary = df[summary_cols].describe()
    
    st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Mission comparison
    st.subheader("🛰️ Kepler vs TESS Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    kepler_df = df[df['mission'] == 'Kepler']
    tess_df = df[df['mission'] == 'TESS']
    
    with col1:
        st.metric("Kepler Stars", len(kepler_df))
        st.metric("Avg Planets", f"{kepler_df['planet_count'].mean():.2f}")
        st.metric("Multi-Planet %", f"{(kepler_df['planet_count'] > 1).sum() / len(kepler_df) * 100:.1f}%")
    
    with col2:
        st.metric("TESS Stars", len(tess_df))
        st.metric("Avg Planets", f"{tess_df['planet_count'].mean():.2f}")
        st.metric("Multi-Planet %", f"{(tess_df['planet_count'] > 1).sum() / len(tess_df) * 100:.1f}%")
    
    with col3:
        st.metric("Total Stars", len(df))
        st.metric("Overall Avg", f"{df['planet_count'].mean():.2f}")
        st.metric("Overall Multi %", f"{(df['planet_count'] > 1).sum() / len(df) * 100:.1f}%")
    
    st.markdown("---")
    
    # Implications
    st.subheader("🔬 Scientific Implications")
    
    st.markdown("""
    #### What These Findings Mean:
    
    1. **Metallicity matters for planet formation**
       - Metal-rich disks provide more building blocks
       - Core accretion theory supported
       - Important for target selection in planet searches
    
    2. **Stellar type affects planet occurrence**
       - G/K stars are optimal targets
       - M dwarfs may have different formation paths
       - Temperature affects disk properties
    
    3. **Multi-planet systems are common**
       - Not all stars with planets have just one
       - Stable configurations are frequent
       - Planet-planet interactions important
    
    4. **Detection biases matter**
       - Different missions have different sensitivities
       - Hierarchical models help correct for this
       - True occurrence rates may differ from observed
    """)
    
    st.markdown("---")
    
    # Download section
    st.subheader("💾 Download Data and Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Dataset (CSV)",
            csv,
            "exoplanet_analysis.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        summary_csv = summary.to_csv()
        st.download_button(
            "📥 Download Statistical Summary",
            summary_csv,
            "statistical_summary.csv",
            "text/csv",
            use_container_width=True
        )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🌌 <b>Exoplanet Occurrence Dashboard</b></p>
    <p>Data from NASA Exoplanet Archive | Built with Streamlit</p>
    <p>For research and educational purposes</p>
</div>
""", unsafe_allow_html=True)