import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('AQI.csv')
st.set_page_config(layout='wide')
selectbox_vals = 'AQI Value','CO AQI Value','Ozone AQI Value','NO2 AQI Value', 'PM2.5 AQI Value'

# Create an instance of LabelEncoder
encoder = LabelEncoder()

# Fit and transform the column
df['AQI_cat_enc'] = encoder.fit_transform(df['AQI Category'])
df['CO_cat_enc'] = encoder.fit_transform(df['CO AQI Category'])
df['Ozone_cat_enc'] = encoder.fit_transform(df['Ozone AQI Category'])
df['NO2_cat_enc'] = encoder.fit_transform(df['NO2 AQI Category'])
df['PM2.5_cat_enc'] = encoder.fit_transform(df['PM2.5 AQI Category'])

# Good : 1, prev:0
# Moderate : 2, prev:2
# Unhealthy for Sensitive Groups: 3, prev: 4
# Unhealthy : 4, prev: 3
# Very Unhealthy : 5, prev: 5
# Hazardous : 6, prev: 1
## Replace the original with the values above

replacement_dict = {0: 1, 4: 3, 3: 4, 1:6}
df[['AQI_cat_enc','CO_cat_enc','Ozone_cat_enc','NO2_cat_enc','PM2.5_cat_enc']] = df[['AQI_cat_enc','CO_cat_enc','Ozone_cat_enc','NO2_cat_enc','PM2.5_cat_enc']].replace(replacement_dict)

# multioutput_classifier is a good choice but i need to be able to calculate whats what 
# for every single entry hence 5 models will be required
X1 = df[['AQI Value']]
Y1 = df['AQI_cat_enc']

X2 = df[['CO AQI Value']]
Y2 = df['CO_cat_enc']

X3 = df[['Ozone AQI Value']]
Y3 = df['Ozone_cat_enc']

X4 = df[['NO2 AQI Value']]
Y4 = df['NO2_cat_enc']

X5 = df[['PM2.5 AQI Value']]
Y5 = df['PM2.5_cat_enc']

# setting classifier
dlc1 = DecisionTreeClassifier()
dlc2 = DecisionTreeClassifier()
dlc3 = DecisionTreeClassifier()
dlc4 = DecisionTreeClassifier()
dlc5 = DecisionTreeClassifier()

# fitting models
model1 = dlc1.fit(X1,Y1)
model2 = dlc2.fit(X2,Y2)
model3 = dlc3.fit(X3,Y3)
model4 = dlc4.fit(X4,Y4)
model5 = dlc5.fit(X5,Y5)

st.header('AQI Analysis & Affect Asessment')
with st.container():
  col1, col2, col3 = st.columns((1,3,1))

with col2:
  st.write('#### AQI Plotting:')
  select_box = st.selectbox('Choose the cateogry to plot:',selectbox_vals)

  if select_box == 'AQI Value':
    colors = 'tealrose'
  elif select_box == 'PM2.5 AQI Value':
    colors = 'tealrose'
  elif select_box == 'Ozone AQI Value':
    colors = 'balance'
  elif select_box == 'NO2 AQI Value':
    colors = 'spectral'
  else:
    colors = 'magma'


  fig = px.scatter_mapbox(
    data_frame=df,
    lat='lat',  # Column name for latitude values
    lon='lng',  # Column name for longitude values
    color=select_box,
    mapbox_style='dark',
    hover_name='City',
    hover_data=[select_box, 'Country'],
    zoom=1,
    color_continuous_scale=colors,
    opacity=0.9
)

  fig.update_layout(mapbox_style='open-street-map'
                  ,width=850,height=600,mapbox=dict(center=dict(lat=13, lon=32)))

  st.plotly_chart(fig)

  with col1:
    st.write('#### City Specific Info:')

    select_box_country = st.selectbox('Country Name:',df['Country'].unique())
    select_box_city = st.selectbox('City Name:',df['City'].loc[df['Country']==select_box_country].unique())
    
    df2 = pd.DataFrame(df.loc[df['City'] == select_box_city])
    
    last_row = df2.iloc[-1]
    df3 = pd.DataFrame({'AQI': [last_row['AQI Value'],last_row['AQI Category']],
    'CO AQI': [last_row['CO AQI Value'],last_row['CO AQI Category']],
    'Ozone AQI': [last_row['Ozone AQI Value'],last_row['Ozone AQI Category']],
    'NO2 AQI': [last_row['NO2 AQI Value'],last_row['NO2 AQI Category']],
    'PM2.5 AQI': [last_row['PM2.5 AQI Value'],last_row['PM2.5 AQI Category']]}).T
    df3 = df3.rename(columns={0: 'Values', 1: 'Category'})
    st.dataframe(df3,use_container_width=True)

    st.bar_chart(df3['Values'],height=250)

    # [['AQI Value','CO AQI Value','Ozone AQI Value','NO2 AQI Value', 'PM2.5 AQI Value']]
    # [['AQI Category','CO AQI Category','Ozone AQI Category','NO2 AQI Category','PM2.5 AQI Category']]
    # use column name change and also reset index to join also search chatgpt for a easier option

    def health_level(i):
      if i == 1:
        st.write('Good')
      elif i == 2:
        st.write('Moderate')
      elif i == 3:
        st.write('Unhealthy for Sensitive Groups')
      elif i == 4:
        st.write('Unhealthy')
      elif i == 5:
        st.write('Extremely Unhealthy')
      elif i == 6:
        st.write('Hazardous')
      else:
        st.write('input invalid')
    
    

  with col3: 
    st.write('#### Affect Asessment:') 
    aqi_input = st.number_input('Insert AQI value:')
    aqi_input = [[aqi_input]]
    aqi_pred = model1.predict(aqi_input)
    health_level(aqi_pred)

    
    co_input = st.number_input('Insert CO AQI value:')
    co_input =[[co_input]]
    co_pred = model2.predict(co_input)
    health_level(co_pred)

    ozone_input = st.number_input('Insert Ozone AQI value:')
    ozone_input = [[ozone_input]]
    ozone_pred = model3.predict(ozone_input)
    health_level(ozone_pred)

    no2_input = st.number_input('Insert NO2 AQI value:')
    no2_input = [[no2_input]]
    no2_pred = model4.predict(no2_input)
    health_level(no2_pred)

    pm_input = st.number_input('Insert PM2.5 AQI value:')
    pm_input = [[pm_input]]
    pm_pred = model5.predict(pm_input)
    health_level(pm_pred)
    



