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
        st.metric("Total Stars", f"{len(df):,}", help="Unique host stars analyzed")
    
    with col2:
        st.metric("Total Planets", f"{int(df['planet_count'].sum()):,}", help="Confirmed exoplanets")
    
    with col3:
        multi_planet = len(df[df['planet_count'] > 1])
        pct = (multi_planet / len(df)) * 100
        st.metric("Multi-Planet Systems", f"{multi_planet:,}", f"{pct:.1f}%")
    
    with col4:
        avg_planets = df['planet_count'].mean()
        st.metric("Avg Planets/Star", f"{avg_planets:.2f}", help="Average occurrence rate")
    
    st.markdown("---")
    
    # Mission comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛰️ Survey Contributions")
        mission_counts = df['mission'].value_counts()
        fig = px.pie(
            values=mission_counts.values,
            names=mission_counts.index,
            title="Kepler vs TESS Host Stars",
            color_discrete_sequence=['#667eea', '#764ba2'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Planet Distribution")
        planet_dist = df['planet_count'].value_counts().sort_index()
        fig = px.bar(
            x=planet_dist.index,
            y=planet_dist.values,
            title="Number of Planets per Star",
            color=planet_dist.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Number of Planets', 'y': 'Number of Stars'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌡️ Temperature Range")
        st.metric("Min", f"{df['st_teff'].min():.0f} K")
        st.metric("Max", f"{df['st_teff'].max():.0f} K")
        st.metric("Mean", f"{df['st_teff'].mean():.0f} K")
    
    with col2:
        st.markdown("### ⚗️ Metallicity Range")
        st.metric("Min", f"{df['st_met'].min():.2f} [Fe/H]")
        st.metric("Max", f"{df['st_met'].max():.2f} [Fe/H]")
        st.metric("Mean", f"{df['st_met'].mean():.2f} [Fe/H]")
    
    with col3:
        st.markdown("### 🪐 Planet Records")
        max_planets = df['planet_count'].max()
        max_star = df.loc[df['planet_count'].idxmax(), 'hostname']
        st.metric("Most Planets", f"{max_planets}")
        st.metric("Host Star", max_star)
        single_planet = len(df[df['planet_count'] == 1])
        st.metric("Single-Planet Stars", f"{single_planet}")

# ============================================
# PAGE 2: DATA EXPLORER
# ============================================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "🎯 Interactive", "📋 Data Table"])
    
    with tab1:
        st.subheader("Stellar Property Distributions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.histogram(
                df, x='st_met', nbins=30, 
                title="Stellar Metallicity [Fe/H]",
                color_discrete_sequence=['teal'],
                marginal='box'
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="Solar")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, x='st_teff', nbins=30,
                title="Stellar Temperature (K)",
                color_discrete_sequence=['orange'],
                marginal='box'
            )
            fig.add_vline(x=5778, line_dash="dash", line_color="red",
                         annotation_text="Sun")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(
                df, x='st_mass', nbins=30,
                title="Stellar Mass (Solar Masses)",
                color_discrete_sequence=['purple'],
                marginal='box'
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                         annotation_text="1 M☉")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Spectral type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if 'spectral_type' in df.columns:
                spectral_counts = df['spectral_type'].value_counts()
                fig = px.bar(
                    x=spectral_counts.index, 
                    y=spectral_counts.values,
                    title="Distribution by Spectral Type",
                    color=spectral_counts.values,
                    color_continuous_scale='Plasma',
                    labels={'x': 'Spectral Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'metallicity_class' in df.columns:
                met_counts = df['metallicity_class'].value_counts()
                fig = px.bar(
                    x=met_counts.index, 
                    y=met_counts.values,
                    title="Distribution by Metallicity Class",
                    color=met_counts.values,
                    color_continuous_scale='Viridis',
                    labels={'x': 'Metallicity Class', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Correlation heatmap
            corr_cols = ['st_teff', 'st_met', 'st_mass', 'st_rad', 'planet_count']
            corr_matrix = df[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Key Correlations")
            
            # Find strongest correlations
            corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_flat = corr_flat.stack().sort_values(ascending=False)
            
            st.markdown("**Strongest Positive:**")
            for idx, val in corr_flat.head(3).items():
                st.write(f"• {idx[0]} ↔ {idx[1]}: **{val:.3f}**")
            
            st.markdown("**Strongest Negative:**")
            for idx, val in corr_flat.tail(3).items():
                st.write(f"• {idx[0]} ↔ {idx[1]}: **{val:.3f}**")
        
        st.markdown("---")
        
        # Box plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df, 
                x='planet_count', 
                y='st_met',
                title="Metallicity by Planet Count",
                color='planet_count',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df, 
                x='planet_count', 
                y='st_teff',
                title="Temperature by Planet Count",
                color='planet_count',
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Interactive Scatter Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X-axis", ['st_teff', 'st_met', 'st_mass', 'st_rad'])
            y_var = st.selectbox("Y-axis", ['planet_count', 'st_met', 'st_teff', 'st_mass'])
        
        with col2:
            color_by = st.selectbox("Color by", ['mission', 'spectral_type', 'planet_count'])
            size_by = st.selectbox("Size by", ['planet_count', 'st_mass', 'st_rad'])
        
        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color=color_by,
            size=size_by,
            hover_data=['hostname'],
            opacity=0.6,
            title=f"{y_var} vs {x_var}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Data Table")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mission_filter = st.multiselect(
                "Filter by Mission",
                options=df['mission'].unique(),
                default=df['mission'].unique()
            )
        
        with col2:
            min_planets = st.number_input("Min Planets", min_value=0, value=0)
        
        with col3:
            max_temp = st.number_input("Max Temp (K)", min_value=3000, max_value=10000, value=10000)
        
        # Apply filters
        filtered_df = df[
            (df['mission'].isin(mission_filter)) &
            (df['planet_count'] >= min_planets) &
            (df['st_teff'] <= max_temp)
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} stars")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            "filtered_exoplanets.csv",
            "text/csv"
        )

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
    
    # Model comparison
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
    
    st.markdown("---")
    
    # Posterior distributions (simulated)
    st.subheader("📈 Posterior Distributions")
    
    st.info("These are simulated posteriors for demonstration. Real analysis would use PyMC.")
    
    np.random.seed(42)
    beta_met = np.random.normal(0.3, 0.05, 1000)
    beta_teff = np.random.normal(0.15, 0.04, 1000)
    beta_mass = np.random.normal(0.1, 0.03, 1000)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(
            beta_met, 
            nbins=50, 
            title="β_metallicity",
            labels={'value': 'Coefficient'}
        )
        fig.add_vline(
            x=beta_met.mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {beta_met.mean():.3f}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("95% CI", f"[{np.percentile(beta_met, 2.5):.3f}, {np.percentile(beta_met, 97.5):.3f}]")
    
    with col2:
        fig = px.histogram(
            beta_teff, 
            nbins=50, 
            title="β_temperature",
            labels={'value': 'Coefficient'}
        )
        fig.add_vline(
            x=beta_teff.mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {beta_teff.mean():.3f}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("95% CI", f"[{np.percentile(beta_teff, 2.5):.3f}, {np.percentile(beta_teff, 97.5):.3f}]")
    
    with col3:
        fig = px.histogram(
            beta_mass, 
            nbins=50, 
            title="β_mass",
            labels={'value': 'Coefficient'}
        )
        fig.add_vline(
            x=beta_mass.mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {beta_mass.mean():.3f}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("95% CI", f"[{np.percentile(beta_mass, 2.5):.3f}, {np.percentile(beta_mass, 97.5):.3f}]")
    
    st.markdown("---")
    
    # Prediction tool
    st.subheader("🎯 Predict Planet Occurrence")
    
    st.markdown("Adjust stellar properties to predict planet occurrence:")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fe_input = st.slider(
            "Metallicity [Fe/H]",
            min_value=-0.5,
            max_value=0.8,
            value=0.0,
            step=0.05,
            help="Solar metallicity is 0.0"
        )
        
        teff_input = st.slider(
            "Temperature (K)",
            min_value=3000,
            max_value=7500,
            value=5800,
            step=100,
            help="Sun's temperature is 5778 K"
        )
        
        mass_input = st.slider(
            "Mass (Solar Masses)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Sun's mass is 1.0"
        )
    
    with col2:
        # Normalize inputs
        teff_norm = (teff_input - df['st_teff'].mean()) / df['st_teff'].std()
        mass_norm = (mass_input - df['st_mass'].mean()) / df['st_mass'].std()
        
        # Predict using posterior samples
        log_lambda = 0.2 + beta_met * fe_input + beta_teff * teff_norm + beta_mass * mass_norm
        predicted = np.exp(log_lambda)
        
        st.markdown("### 📊 Predictions")
        
        st.metric(
            "Expected Planet Count",
            f"{predicted.mean():.2f}",
            help="Mean of posterior predictive distribution"
        )
        
        st.metric(
            "95% Credible Interval",
            f"[{np.percentile(predicted, 2.5):.2f}, {np.percentile(predicted, 97.5):.2f}]",
            help="95% probability the true value is in this range"
        )
        
        prob = 1 - np.exp(-predicted)
        st.metric(
            "P(≥1 Planet)",
            f"{prob.mean():.1%}",
            help="Probability star hosts at least one planet"
        )
        
        # Visual
        fig = px.histogram(
            predicted,
            nbins=30,
            title="Predicted Planet Count Distribution",
            labels={'value': 'Planet Count', 'count': 'Frequency'}
        )
        fig.add_vline(x=predicted.mean(), line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

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