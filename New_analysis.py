import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from colorama import Fore, Style, init

# Check if running in Streamlit environment
IS_STREAMLIT = "streamlit" in sys.modules

if IS_STREAMLIT:
    import streamlit as st
    init(autoreset=True)
    warnings.filterwarnings('ignore')
    st.set_page_config(page_title="Net Zero Energy Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("🌍 Net Zero Energy Dashboard")
    st.markdown("<style>body {background-color: #f0f2f6;}</style>", unsafe_allow_html=True)
    st.markdown("---")

# Load data with validation
@st.cache_data if IS_STREAMLIT else lambda x: x
def load_data():
    try:
        data = pd.read_csv('hpcldata.csv')
        required_columns = ['Date', 'State', 'Energy_Source', 'Consumption_TWh', 'CO2_Emissions_MT', 
                          'Renewable_Percentage', 'Population_Million', 'GDP_Billion_USD']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            if IS_STREAMLIT:
                st.error(f"Missing columns in hpcldata.csv: {missing_columns}. Please check the file.")
            return pd.DataFrame()
        if IS_STREAMLIT:
            st.success("Data loaded successfully!")
        return data
    except FileNotFoundError:
        if IS_STREAMLIT:
            st.error("Error: 'hpcldata.csv' not found. Please place it in the same directory and restart.")
        else:
            print("Error: 'hpcldata.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        if IS_STREAMLIT:
            st.error(f"Error loading data: {str(e)}")
        else:
            print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

data = load_data()

if IS_STREAMLIT and not data.empty:
    # Sidebar for filters
    st.sidebar.header("🔧 Filters")
    states = ['All'] + sorted(data['State'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", options=states, key="state_filter")
    selected_source = st.sidebar.selectbox("Select Energy Source", options=['All'] + sorted(data['Energy_Source'].dropna().unique().tolist()), key="source_filter")
    if st.sidebar.button("Reset Filters", key="reset_filters"):
        selected_state = 'All'
        selected_source = 'All'

    # Filter data
    filtered_data = data.copy()
    if selected_state != 'All':
        filtered_data = filtered_data[filtered_data['State'] == selected_state]
    if selected_source != 'All':
        filtered_data = filtered_data[filtered_data['Energy_Source'] == selected_source]

    # Data Preparation (hidden)
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], errors='coerce')
    renewable_sources = ['Solar', 'Wind', 'Hydro']
    filtered_data['Is_Renewable'] = filtered_data['Energy_Source'].isin(renewable_sources)
    filtered_data['Carbon_Intensity'] = filtered_data.apply(
        lambda row: row['CO2_Emissions_MT'] / row['Consumption_TWh'] if row['Consumption_TWh'] != 0 else np.nan, axis=1
    )

    # Current Energy Landscape
    st.header("📊 Current Energy Landscape")
    col1, col2 = st.columns(2)
    with col1:
        total_consumption = filtered_data['Consumption_TWh'].sum()
        total_emissions = filtered_data['CO2_Emissions_MT'].sum()
        st.metric("Total Consumption", f"{total_consumption:.1f} TWh")
        st.metric("Total CO2 Emissions", f"{total_emissions:.1f} MT")
    with col2:
        renewable_consumption = filtered_data[filtered_data['Is_Renewable']]['Consumption_TWh'].sum()
        renewable_percentage = (renewable_consumption / total_consumption * 100) if total_consumption > 0 else 0
        st.metric("Renewable Consumption", f"{renewable_consumption:.1f} TWh")
        st.metric("Renewable Percentage", f"{renewable_percentage:.1f}%")

    # State-wise Insights
    st.header("🌍 State-wise Insights")
    state_analysis = filtered_data.groupby('State').agg({
        'Consumption_TWh': 'sum',
        'CO2_Emissions_MT': 'sum',
        'Renewable_Percentage': 'mean',
        'Population_Million': 'first',
        'GDP_Billion_USD': 'first'
    }).round(2).fillna(0)
    if not state_analysis.empty:
        st.dataframe(state_analysis.style.background_gradient(cmap='Greens').set_properties(**{'text-align': 'center'}))
    else:
        st.warning("No data available for state-wise analysis with current filters.")

    # Net Zero Scenarios
    st.header("🎯 Net Zero Scenarios")
    scenarios = {
        '2030 Target': {'renewable_pct': 50, 'year': 2030},
        '2040 Target': {'renewable_pct': 80, 'year': 2040},
        '2050 Net Zero': {'renewable_pct': 100, 'year': 2050}
    }
    scenario_results = []
    for name, params in scenarios.items():
        target_pct = params['renewable_pct']
        target_consumption = (target_pct / 100) * total_consumption
        additional_renewable = max(0, target_consumption - renewable_consumption)
        fossil_emissions = filtered_data[~filtered_data['Is_Renewable']]['CO2_Emissions_MT'].sum()
        emissions_reduction = (fossil_emissions * (target_pct - renewable_percentage) / (100 - renewable_percentage)
                             if renewable_percentage < 100 and fossil_emissions > 0 else 0)
        investment_needed = additional_renewable * 2.5
        scenario_results.append({
            'Scenario': name, 'Target_Year': params['year'], 'Renewable_Pct': target_pct,
            'Additional_TWh': additional_renewable, 'Emissions_Reduction_MT': emissions_reduction,
            'Investment_Needed_Billion': investment_needed
        })
    scenario_df = pd.DataFrame(scenario_results)
    if not scenario_df.empty:
        st.dataframe(scenario_df.style.highlight_max(axis=0).set_properties(**{'text-align': 'center'}))
    else:
        st.warning("No data available for scenario analysis.")

    # Visualizations with Chart Type Switcher
    st.header("📈 Energy Visualizations")
    chart_types = ['Bar', 'Pie', 'Line']
    selected_chart_type = st.selectbox("Switch Chart Type", options=chart_types, key="chart_type")

    tabs = st.tabs(["Energy Mix", "CO2 Emissions", "State Emissions", "Renewable Split", "Carbon Intensity", "Timeline"])

    with tabs[0]:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        energy_mix = filtered_data.groupby('Energy_Source')['Consumption_TWh'].sum()
        if not energy_mix.empty:
            if selected_chart_type == 'Bar':
                energy_mix.plot(kind='bar', ax=ax1)
                ax1.set_title('Energy Mix by Source')
            elif selected_chart_type == 'Pie':
                ax1.pie(energy_mix, labels=energy_mix.index, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Energy Mix Distribution')
            else:  # Line
                ax1.plot(energy_mix.index, energy_mix.values)
                ax1.set_title('Energy Mix Trend')
            st.pyplot(fig1)
        else:
            st.warning("No data available for Energy Mix visualization.")

    with tabs[1]:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        emissions_by_source = filtered_data.groupby('Energy_Source')['CO2_Emissions_MT'].sum()
        if not emissions_by_source.empty:
            if selected_chart_type == 'Bar':
                emissions_by_source.plot(kind='bar', ax=ax2)
                ax2.set_title('CO2 Emissions by Source')
            elif selected_chart_type == 'Pie':
                ax2.pie(emissions_by_source, labels=emissions_by_source.index, autopct='%1.1f%%')
                ax2.set_title('CO2 Emissions Distribution')
            else:  # Line
                ax2.plot(emissions_by_source.index, emissions_by_source.values)
                ax2.set_title('CO2 Emissions Trend')
            st.pyplot(fig2)
        else:
            st.warning("No data available for CO2 Emissions visualization.")

    with tabs[2]:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        state_emissions = filtered_data.groupby('State')['CO2_Emissions_MT'].sum()
        if not state_emissions.empty:
            if selected_chart_type == 'Bar':
                state_emissions.plot(kind='bar', ax=ax3)
                ax3.set_title('CO2 Emissions by State')
            elif selected_chart_type == 'Pie':
                ax3.pie(state_emissions, labels=state_emissions.index, autopct='%1.1f%%')
                ax3.set_title('CO2 Emissions Distribution by State')
            else:  # Line
                ax3.plot(state_emissions.index, state_emissions.values)
                ax3.set_title('CO2 Emissions Trend by State')
            st.pyplot(fig3)
        else:
            st.warning("No data available for State Emissions visualization.")

    with tabs[3]:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        renewable_total = filtered_data[filtered_data['Is_Renewable']]['Consumption_TWh'].sum()
        non_renewable_total = filtered_data[~filtered_data['Is_Renewable']]['Consumption_TWh'].sum()
        data_to_plot = [renewable_total, non_renewable_total]
        labels = ['Renewable', 'Non-Renewable']
        if any(data_to_plot):
            if selected_chart_type == 'Bar':
                ax4.bar(labels, data_to_plot)
                ax4.set_title('Renewable vs Non-Renewable Consumption')
            elif selected_chart_type == 'Pie':
                ax4.pie(data_to_plot, labels=labels, autopct='%1.1f%%')
                ax4.set_title('Renewable vs Non-Renewable Distribution')
            else:  # Line
                ax4.plot(labels, data_to_plot)
                ax4.set_title('Renewable vs Non-Renewable Trend')
            st.pyplot(fig4)
        else:
            st.warning("No data available for Renewable Split visualization.")

    with tabs[4]:
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        carbon_intensity = filtered_data.groupby('Energy_Source')['Carbon_Intensity'].mean()
        if not carbon_intensity.empty:
            if selected_chart_type == 'Bar':
                carbon_intensity.plot(kind='bar', ax=ax5)
                ax5.set_title('Carbon Intensity by Source')
            elif selected_chart_type == 'Pie':
                ax5.pie(carbon_intensity, labels=carbon_intensity.index, autopct='%1.1f%%')
                ax5.set_title('Carbon Intensity Distribution')
            else:  # Line
                ax5.plot(carbon_intensity.index, carbon_intensity.values)
                ax5.set_title('Carbon Intensity Trend')
            st.pyplot(fig5)
        else:
            st.warning("No data available for Carbon Intensity visualization.")

    with tabs[5]:
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        timeline_data = filtered_data.groupby(['Date', 'State'])['Consumption_TWh'].sum().unstack(fill_value=0)
        if not timeline_data.empty:
            if selected_chart_type == 'Bar':
                timeline_data.plot(kind='bar', ax=ax6)
                ax6.set_title('Energy Consumption Timeline by State')
            elif selected_chart_type == 'Pie':
                for col in timeline_data.columns:
                    ax6.pie([timeline_data[col].sum()], labels=[col], autopct='%1.1f%%')
                    ax6.set_title(f'Consumption for {col}')
                    st.pyplot(fig6)
                    fig6, ax6 = plt.subplots(figsize=(12, 6))
            else:  # Line
                for column in timeline_data.columns:
                    ax6.plot(timeline_data.index, timeline_data[column], label=column)
                ax6.legend()
                ax6.set_title('Energy Consumption Timeline')
            st.pyplot(fig6)
        else:
            st.warning("No data available for Timeline visualization.")

    # Export Data
    st.header("📥 Export Data")
    filtered_data.to_csv('enhanced_energy_analysis.csv', index=False)
    scenario_df.to_csv('net_zero_scenarios.csv', index=False)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(label="Download Enhanced Data", data=open('enhanced_energy_analysis.csv', 'rb'), file_name='enhanced_energy_analysis.csv', key="download1")
    with col2:
        st.download_button(label="Download Scenarios", data=open('net_zero_scenarios.csv', 'rb'), file_name='net_zero_scenarios.csv', key="download2")

    st.markdown("---")
    st.markdown("**Note:** The data utilized in this dashboard is sample data sourced for illustrative purposes, designed to provide insights into potential energy trends and scenarios.")

else:
    if not data.empty:
        print("Data loaded successfully. Please run with 'streamlit run complete_analysis.py' to view the app.")
    else:
        print("This script is designed to run with 'streamlit run complete_analysis.py'. Please ensure 'hpcldata.csv' is available.")