import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import numpy as np

soccer_data = pd.read_csv('SalaryPrediction.csv')
nba_data = pd.read_csv('nba_2022-23_all_stats_with_salary.csv')

soccer_data_cleaned = soccer_data.dropna()
nba_data_cleaned = nba_data.dropna()
nba_data_cleaned = nba_data[nba_data['Team'].apply(lambda x: len(str(x)) <= 3)]

soccer_data_cleaned['Wage'] = soccer_data_cleaned['Wage'].replace({',': ''}, regex=True).astype(float)
nba_data_cleaned['Salary'] = nba_data_cleaned['Salary'].replace({',': ''}, regex=True).astype(float)

soccer_label_encoders = {}
for column in ['Club', 'League', 'Nation', 'Position']:
    le = LabelEncoder()
    soccer_data_cleaned[column] = le.fit_transform(soccer_data_cleaned[column])
    soccer_label_encoders[column] = le

X_soccer = soccer_data_cleaned.drop(columns=['Wage'])
y_soccer = soccer_data_cleaned['Wage']

soccer_scaler = StandardScaler()
X_soccer[['Age', 'Apps', 'Caps']] = soccer_scaler.fit_transform(X_soccer[['Age', 'Apps', 'Caps']])

X_soccer_train, X_soccer_test, y_soccer_train, y_soccer_test = train_test_split(X_soccer, y_soccer, test_size=0.2, random_state=42)

rf_soccer = RandomForestRegressor(n_estimators=100, random_state=42)
rf_soccer.fit(X_soccer_train, y_soccer_train)

nba_data_cleaned = nba_data_cleaned.drop(columns=[
    'Unnamed: 0', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 
    'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 
    'PTS', 'Total Minutes', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
    'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 
    'DBPM', 'BPM', 'VORP'
])

nba_label_encoders = {}
for column in ['Position', 'Team']:
    le = LabelEncoder()
    nba_data_cleaned[column] = le.fit_transform(nba_data_cleaned[column])
    nba_label_encoders[column] = le

X_nba = nba_data_cleaned.drop(columns=['Player Name', 'Salary', 'FG'])
y_nba = nba_data_cleaned['Salary']

nba_scaler = StandardScaler()
X_nba[['Age', 'GP', 'GS', 'MP']] = nba_scaler.fit_transform(X_nba[['Age', 'GP', 'GS', 'MP']])

X_nba_train, X_nba_test, y_nba_train, y_nba_test = train_test_split(X_nba, y_nba, test_size=0.2, random_state=42)

rf_nba = RandomForestRegressor(n_estimators=100, random_state=42)
rf_nba.fit(X_nba_train, y_nba_train)

st.title("Wage Predictor")

if 'prediction_type' not in st.session_state:
    st.session_state.prediction_type = 'Soccer Wage Predictor'

col1, col2 = st.columns(2)
with col1:
    if st.button('Soccer Wage Predictor'):
        st.session_state.prediction_type = 'Soccer Wage Predictor'
with col2:
    if st.button('NBA Wage Predictor'):
        st.session_state.prediction_type = 'NBA Wage Predictor'

if st.session_state.prediction_type == 'Soccer Wage Predictor':
    st.subheader("Soccer Wage Prediction")
    selected_league = st.selectbox('Select League', soccer_label_encoders['League'].classes_)
    filtered_soccer_data = soccer_data_cleaned[soccer_data_cleaned['League'] == soccer_label_encoders['League'].transform([selected_league])[0]]
    filtered_clubs = soccer_label_encoders['Club'].inverse_transform(filtered_soccer_data['Club'].unique())
    
    selected_club = st.selectbox('Select Club', filtered_clubs)
    age = st.slider('Age', min_value=int(soccer_data_cleaned['Age'].min()), max_value=int(soccer_data_cleaned['Age'].max()), value=int(soccer_data_cleaned['Age'].mean()))
    nation = st.selectbox('Select Nationality', soccer_label_encoders['Nation'].classes_)
    position = st.selectbox('Select Position', soccer_label_encoders['Position'].classes_)
    apps = st.slider('Apps', min_value=int(soccer_data_cleaned['Apps'].min()), max_value=int(soccer_data_cleaned['Apps'].max()), value=int(soccer_data_cleaned['Apps'].mean()))
    caps = st.slider('Caps', min_value=int(soccer_data_cleaned['Caps'].min()), max_value=int(soccer_data_cleaned['Caps'].max()), value=int(soccer_data_cleaned['Caps'].mean()))
    
    encoded_soccer_values = {
        'League': soccer_label_encoders['League'].transform([selected_league])[0],
        'Club': soccer_label_encoders['Club'].transform([selected_club])[0],
        'Nation': soccer_label_encoders['Nation'].transform([nation])[0],
        'Position': soccer_label_encoders['Position'].transform([position])[0],
        'Age': age,
        'Apps': apps,
        'Caps': caps
    }

    input_soccer_df = pd.DataFrame([encoded_soccer_values])
    input_soccer_df = input_soccer_df[X_soccer.columns] 
    input_soccer_df[['Age', 'Apps', 'Caps']] = soccer_scaler.transform(input_soccer_df[['Age', 'Apps', 'Caps']])
    
    predicted_soccer_wage = rf_soccer.predict(input_soccer_df)
    predicted_soccer_wage = np.maximum(predicted_soccer_wage, 0)  
    
    st.write(f"Predicted Soccer Wage: ${predicted_soccer_wage[0]:,.2f}")

elif st.session_state.prediction_type == 'NBA Wage Predictor':
    st.subheader("NBA Wage Prediction")
    selected_team = st.selectbox('Select NBA Team', nba_label_encoders['Team'].classes_)
    position = st.selectbox('Select Position', nba_label_encoders['Position'].classes_)
    age = st.slider('Age', min_value=int(nba_data_cleaned['Age'].min()), max_value=int(nba_data_cleaned['Age'].max()), value=int(nba_data_cleaned['Age'].mean()))
    gp = st.slider('Games Played (GP)', min_value=int(nba_data_cleaned['GP'].min()), max_value=int(nba_data_cleaned['GP'].max()), value=int(nba_data_cleaned['GP'].mean()))
    gs = st.slider('Games Started (GS)', min_value=int(nba_data_cleaned['GS'].min()), max_value=int(nba_data_cleaned['GS'].max()), value=int(nba_data_cleaned['GS'].mean()))
    mp = st.slider('Minutes per Game (MP)', min_value=float(nba_data_cleaned['MP'].min()), max_value=float(nba_data_cleaned['MP'].max()), value=float(nba_data_cleaned['MP'].mean()))
    
    encoded_nba_values = {
        'Team': nba_label_encoders['Team'].transform([selected_team])[0],
        'Position': nba_label_encoders['Position'].transform([position])[0],
        'Age': age,
        'GP': gp,
        'GS': gs,
        'MP': mp
    }

    input_nba_df = pd.DataFrame([encoded_nba_values])
    input_nba_df = input_nba_df[X_nba.columns] 
    input_nba_df[['Age', 'GP', 'GS', 'MP']] = nba_scaler.transform(input_nba_df[['Age', 'GP', 'GS', 'MP']])
    
    predicted_nba_wage = rf_nba.predict(input_nba_df)
    predicted_nba_wage = np.maximum(predicted_nba_wage, 0)  
    
    st.write(f"Predicted NBA Wage: ${predicted_nba_wage[0]:,.2f}")
