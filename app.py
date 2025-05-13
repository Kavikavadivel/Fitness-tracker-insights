# -- coding: utf-8 --
"""Fitness Tracker Dashboard with Goal Setting and Data Addition"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data function
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\\Users\\kavik\\OneDrive\\文档\fti_proj\\fti ds.csv')
    df['dates'] = pd.to_datetime(df['dates'], errors='coerce')  # This will try to automatically infer the format
    
    return df

# Save data to CSV
def save_data(df, filepath=r'C:\\Users\\kavik\\OneDrive\\文档\fti_proj\\fti ds.csv'):
    df.to_csv(filepath, index=False)

# Load the dataframe
df = load_data()

# Initialize session state for navigation and goals
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if 'goals' not in st.session_state:
    st.session_state.goals = {}

# Function to set the current page
def set_page(page_name):
    st.session_state.page = page_name

# Header with Logo and Title
st.markdown("""
    <div style="text-align: center;">
        <img src="https://plus.unsplash.com/premium_photo-1681433383783-661b519b154a?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Zml0bmVzcyUyMHRyYWNrZXJ8ZW58MHx8MHx8fDA%3D" alt="Logo" style="width:150px;">
        <h1 style="color:#0A74DA;">Fitness Tracker Insights</h1>
        <p>Your data-driven fitness companion</p>
    </div>
    """, unsafe_allow_html=True)

# Navigation Buttons
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,1,1,1,1])
with col1:
    if st.button("🏠 Home"):
        set_page('Home')
with col2:
    if st.button("📈 Data Overview"):
        set_page('Data Overview')
with col3:
    if st.button("📊 Activity Analysis"):
        set_page('Activity Analysis')
with col4:
    if st.button("❤ Health Metrics"):
        set_page('Health Metrics')
with col5:
    if st.button("🛌 Sleep Analysis"):
        set_page('Sleep Analysis')
with col6:
    if st.button("➕ Add Data"):
        set_page('Add Data')
with col7:
    if st.button("🎯 Goal Settings"):
        set_page('Goal Settings')
st.markdown("<hr>", unsafe_allow_html=True)

# Home Page
def home_page():
    st.title("Welcome to Fitness Tracker Insights 🏠")
    st.write("""
    This dashboard provides a comprehensive overview of your fitness data, allowing you to monitor and analyze various health and activity metrics. Navigate through the sections using the buttons above to explore different aspects of your fitness journey.
    """)
    st.balloons()

# Data Overview Page
def data_overview_page():
    st.title("Data Overview 📈")
    st.subheader("Raw Data")
    st.dataframe(df)
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

def activity_analysis_page():
    st.title("Activity Analysis 📊")

    # Steps vs Calories Burned
    st.subheader("Steps Count vs. Calories Burned")
    fig_steps_cal = px.scatter(
        df,
        x='steps_count',
        y='calories_burned',
        hover_data=['dates'],
        labels={'steps_count': 'Steps Count', 'calories_burned': 'Calories Burned'},
        title='Calories Burned vs. Steps Count',
        trendline='ols',  # This requires statsmodels
        color='steps_count',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_steps_cal, use_container_width=True)

    # Activity Trends with Metrics
    st.subheader("Activity Trends")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Daily Steps", f"{int(df['steps_count'].mean()):,}")
        st.metric("Max Steps", f"{int(df['steps_count'].max()):,}")
    with col2:
        st.metric("Average Calories Burned", f"{int(df['calories_burned'].mean()):,}")
        st.metric("Max Calories Burned", f"{int(df['calories_burned'].max()):,}")

# Health Metrics Page
def health_metrics_page():
    st.title("Health Metrics ❤")
    # Heart Rate Trends
    st.subheader("Heart Rate Trends")
    fig_hr = px.line(
        df,
        x='dates',
        y='average_heart_rate',
        labels={'dates': 'Date', 'average_heart_rate': 'Heart Rate (BPM)'},
        title='Average Heart Rate Over Time',
        hover_data={'dates': '|%B %d, %Y'}
    )
    fig_hr.update_traces(mode='lines+markers', line=dict(color='red'))
    st.plotly_chart(fig_hr, use_container_width=True)

    # Hydration Level
    st.subheader("Hydration Level")
    fig_hydration = px.bar(
        df,
        x='dates',
        y='hydration_level',
        labels={'dates': 'Date', 'hydration_level': 'Hydration Level (%)'},
        title='Daily Hydration Levels',
        hover_data=['dates', 'hydration_level'],
        color='hydration_level',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_hydration, use_container_width=True)

    # Temperature vs Heart Rate
    st.subheader("Temperature vs Heart Rate")
    fig_temp_hr = px.scatter(
        df,
        x='average_temperature',
        y='average_heart_rate',
        hover_data=['dates'],
        labels={'average_temperature': 'Temperature (°C)', 'average_heart_rate': 'Heart Rate (BPM)'},
        title='Temperature vs. Heart Rate',
        color='average_temperature',
        color_continuous_scale='Hot'
    )
    st.plotly_chart(fig_temp_hr, use_container_width=True)

# Sleep Analysis Page
def sleep_analysis_page():
    st.title("Sleep Analysis 🛌")

    # Sleep Duration Over Time
    st.subheader("Sleep Duration Over Time")
    fig_sleep_time = px.line(
        df,
        x='dates',
        y='hours_of_sleep',
        labels={'dates': 'Date', 'hours_of_sleep': 'Hours of Sleep'},
        title='Daily Sleep Duration',
        hover_data={'dates': '|%B %d, %Y'}
    )
    fig_sleep_time.update_traces(mode='lines+markers', line=dict(color='purple'))
    st.plotly_chart(fig_sleep_time, use_container_width=True)

    # Sleep vs Activity Level
    st.subheader("Sleep vs Activity Level")
    fig_sleep_activity = px.scatter(
        df,
        x='steps_count',
        y='hours_of_sleep',
        hover_data=['dates'],
        labels={'steps_count': 'Steps Count', 'hours_of_sleep': 'Hours of Sleep'},
        title='Sleep Duration vs. Steps Count',
        trendline='ols',  # This requires statsmodels
        color='hours_of_sleep',
        color_continuous_scale='Sunset'
    )
    st.plotly_chart(fig_sleep_activity, use_container_width=True)

    # Sleep Statistics with Metrics
    st.subheader("Sleep Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Sleep Duration", f"{df['hours_of_sleep'].mean():.1f} hours")
    with col2:
        st.metric("Best Sleep Duration", f"{df['hours_of_sleep'].max():.1f} hours")
    with col3:
        st.metric("Worst Sleep Duration", f"{df['hours_of_sleep'].min():.1f} hours")



# Example model setup for machine learning predictions
# Replace this with a trained model for better accuracy
def train_placeholder_model():
    # Dummy training data
    X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Distances in km
    y_train_calories = np.array([50, 100, 150, 200, 250])  # Calories burned
    y_train_hydration = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # Hydration level in liters
    y_train_heart_rate = np.array([70, 72, 74, 76, 78])  # Average heart rate
    y_train_temp = np.array([36.5, 36.7, 36.9, 37.1, 37.3])  # Average temperature

    models = {
        "calories_burned": LinearRegression().fit(X_train, y_train_calories),
        "hydration_level": LinearRegression().fit(X_train, y_train_hydration),
        "average_heart_rate": LinearRegression().fit(X_train, y_train_heart_rate),
        "average_temperature": LinearRegression().fit(X_train, y_train_temp),
    }
    return models


# Train placeholder models (use actual models in production)
ml_models = train_placeholder_model()


def add_data_page():
    st.title("Add New Data")

    # User input for distance walked and sleep hours
    distance_walked = st.number_input("Enter Distance Walked (in km):", min_value=0.0, step=0.1, key="distance_walked")
    sleep_hours = st.number_input("Enter Sleep Hours (hours):", min_value=0.0, step=0.1, key="sleep_hours")

    # Calculate steps count based on distance (1 km = ~1,250 steps as a rough estimate)
    steps_count = int(distance_walked * 1250)

    # Use ML models to calculate other metrics
    if distance_walked > 0:
        calories_burned = ml_models["calories_burned"].predict([[distance_walked]])[0]
        hydration_level = ml_models["hydration_level"].predict([[distance_walked]])[0]
        average_heart_rate = ml_models["average_heart_rate"].predict([[distance_walked]])[0]
        average_temperature = ml_models["average_temperature"].predict([[distance_walked]])[0]
    else:
        calories_burned = hydration_level = average_heart_rate = average_temperature = 0.0

    # Display calculated and entered values
    st.subheader("Calculated Metrics")
    st.write(f"*Steps Count:* {steps_count}")
    st.write(f"*Calories Burned:* {calories_burned:.2f}")
    st.write(f"*Hydration Level:* {hydration_level:.2f} liters")
    st.write(f"*Average Heart Rate:* {average_heart_rate:.2f} bpm")
    st.write(f"*Average Temperature:* {average_temperature:.2f} °C")
    st.write(f"*Sleep Hours:* {sleep_hours} hours")

    # Add Data Button
    if st.button("Add Data"):
        # Prepare data to append to the dataset
        new_data = {
            "dates": pd.Timestamp.now().strftime('%Y-%m-%d'),  # Current date
            "steps_count": steps_count,
            "calories_burned": calories_burned,
            "hydration_level": hydration_level,
            "average_heart_rate": average_heart_rate,
            "average_temperature": average_temperature,
            "hours_of_sleep": sleep_hours,
        }

        # Add data to the dataset
        new_data_df = pd.DataFrame([new_data])

        # Check if 'df' exists in session state and update
        if 'df' not in st.session_state:
            st.session_state.df = load_data()  # Load data if not already loaded

        # Update the DataFrame in session state
        st.session_state.df = pd.concat([st.session_state.df, new_data_df], ignore_index=True)

        # Save data to CSV
        save_data(st.session_state.df)

        # Show confirmation and display the added data
        st.success("New data successfully added!")
        st.write("### Added Data")
        st.write(new_data_df)


def goal_settings_page():
    st.title("Goal Setting and Tracking")

    # Columns for inputting goals
    col1, col2 = st.columns(2)

    with col1:
        steps_goal = st.number_input("Enter Steps Goal", min_value=0, key="steps_goal")
        calories_goal = st.number_input("Enter Calories Goal", min_value=0, key="calories_goal")
        sleep_goal = st.number_input("Enter Sleep Goal (hours)", min_value=0.0, step=0.1, key="sleep_goal")

    with col2:
        hydration_goal = st.number_input("Enter Hydration Goal (%)", min_value=0.0, step=0.1, key="hydration_goal")
        heart_rate_goal = st.number_input("Enter Average Heart Rate Goal", min_value=0, key="heart_rate_goal")
        temp_goal = st.number_input("Enter Average Temperature Goal (°C)", min_value=0.0, step=0.1, key="temp_goal")

    # Save goals in session state
    if st.button("Set Goals"):
        st.session_state['goals'] = {
            "steps_count": steps_goal,
            "calories_burned": calories_goal,
            "hours_of_sleep": sleep_goal,
            "hydration_level": hydration_goal,
            "average_heart_rate": heart_rate_goal,
            "average_temperature": temp_goal
        }
        st.success("Goals successfully set!")

    # Goal attainment section
    st.header("Check Goal Attainment")
    date_to_check = st.date_input("Select Date to Check Goal Attainment")

    if st.button("Check Attainment"):
        # Convert selected date to a string if the dataset uses date strings
        selected_date_str = pd.to_datetime(date_to_check).strftime('%Y-%m-%d')
        selected_data = df[df['dates'] == selected_date_str]

        if not selected_data.empty:
            st.write(f"Data for {selected_date_str}:")
            st.write(selected_data)

            attainment_results = []  # To store goal attainment results

            # Check each goal
            for metric, goal in st.session_state.get('goals', {}).items():
                actual_value = selected_data.iloc[0][metric]
                attained = actual_value >= goal
                attainment_results.append({
                    'metric': metric,
                    'actual_value': actual_value,
                    'goal': goal,
                    'attained': attained
                })

                # Display individual metric attainment
                status = "Attained 🎉" if attained else "Not Attained 😞"
                st.write(f"{metric.replace('_', ' ').capitalize()}: {status}")
                st.write(f"  - Actual: {actual_value}")
                st.write(f"  - Goal: {goal}")

            # Display summary
            st.write("### Goal Attainment Summary")
            for result in attainment_results:
                st.write(f"{result['metric'].capitalize()}: {'✅' if result['attained'] else '❌'}")
        else:
            st.error("No data available for the selected date.")



# Mapping pages to their respective functions
page_functions = {
    'Home': home_page,
    'Data Overview': data_overview_page,
    'Activity Analysis': activity_analysis_page,
    'Health Metrics': health_metrics_page,
    'Sleep Analysis': sleep_analysis_page,
    'Add Data': add_data_page,
    'Goal Settings': goal_settings_page
}

# Display the selected page
if st.session_state.page in page_functions:
    page_functions[st.session_state.page]()
else:
    home_page()