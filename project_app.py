import pandas as pd
from requests import options
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

# make containers
header=st.container()
data_set=st.container()
features=st.container()
plot=st.container()
model_training=st.container()

with header:
    st.title('Project Description')
    st.subheader('General Overview:')
    st.markdown(""" 
                
               - People express their character identification in unique ways and the most important manner is apparel. Someone’s self-notion is pondered via his clothing and brand preferences, and it indicates how a person would like to be.
               - In the last few decades many national and international brands have evolved in Pakistan.
               - Entrepreneurs make use of brands as the principal factor of differentiation to the advantage of aggressive benefit on different competition, gambling being an imperative function in the triumph of companies[1].
               - Clothes today are made from a wide range of different materials. Traditional materials such as cotton, linen and leather are still sourced from plants and animals[2].  
               
               
               References:
                ---
               [1] Kamran, A., Dawood, M. U., Rafi, S. K., Butt, F. M., & Akhtar, K. (2020). Impact of Brand Name on Purchase Intention: A Study on Clothing in Karachi, Pakistan. International Journal of Innovation, Creativity and Change, 278-293. 
               [2] Objective, C. (2021, December 10). What Are Our Clothes Made From? Retrieved from https://www.commonobjective.co/article/what-are-our-clothes-made-from            
    """)
    
# Lets upload the image
    st.image('brand.jpg',width=700)
    
    
with data_set:
# import data
    df=pd.read_csv('df1.csv')
    st.header("A glance at the data set")
    st.write(df)
#######################################################################

with plot:
    dff = pd.read_csv('main_data.csv')
    df1=dff.copy()
    st.header("Data Visualization")
    
    # plot 1    
    df1['years'] = pd.DatetimeIndex(df1['BillDate']).year
    sales_region = (
        df1.groupby(["RegionName", "years"])["NetAmount"]
        .sum()
        .reset_index(name="NetAmount")
    )
    fig = px.bar(
        sales_region,
        x="RegionName",
        y="NetAmount",
        color="years",
        barmode="group",
        title="Yearly Income of Each Region"
    )
    st.write(fig)
         
    
    
    
    
    
    
    
    # plot 2
    fig=px.bar(df1, x='BillMonth', y='SaleExclGST', title='Profit of Each Month')
    fig.update_traces(marker_color='Black')
    fig.update_traces(marker_color='rgb(0, 0, 78)',marker_line_color='rgb(0, 56, 5)')
    st.write(fig)
        
    # plot 3
    # To show year wise comparison with the net amount of sales, we split the Bill Date to years
    #df1['years'] = pd.DatetimeIndex(df1['BillDate']).year
    #To show the total income of each Day on yearly basis for 2016, 2017 & 2018
    df1['BillDate'] = pd.to_datetime(df1['BillDate'], format='%Y/%m/%d')
    dff = df1.groupby(['years','BillDate'])['NetAmount'].sum().reset_index(name='NetAmount')
    fig = px.line(dff, x="BillDate", y="NetAmount", color='years', title='Total Monthly Income of Each Year')
    st.write(fig)
        
    # plot 4
    fig = px.bar(df1, x="years", y="TaxPer", title='Tax Percent with Respect to Each Year')
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = df1.years,
        )
    )
    fig.update_traces(marker_color='rgb(0, 0, 78)',marker_line_color='rgb(0, 56, 5)')
    st.write(fig)
    
    # plot 5
     # Now plotting w.r.t region name and their location wise sales graph
    sales_region = (
        df1.groupby(["RegionName", "years"])["NetAmount"]
        .sum()
        .reset_index(name="NetAmount")
    )
    fig = px.line(
        sales_region,
        x="years",
        y="NetAmount",
        color="RegionName",
        title="Yearly Income of Each Region"
    )
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = df1.years,
        )
    )
    st.write(fig)
        
     # plot 6   
    df_group2 = df1.groupby(['SeasonName','RegionName'])['NetAmount'].sum().reset_index(name='NetAmount')
    fig = px.bar(df_group2, x="RegionName", y="NetAmount", color='SeasonName', title='Season Wise Income of Each Region')
    st.write(fig)
    
    # plot 7
    import calendar
    df1['years'] = pd.DatetimeIndex(df1['BillDate']).year
    df1['BillDate'] = pd.to_datetime(df1['BillDate'], errors='coerce')
    df1['Month'] = df1['BillDate'].dt.month
    df1['Month'] = df1['Month'].apply(lambda x: calendar.month_name[x])
    dff = df1.groupby(['Month','years'])['GST'].sum().reset_index(name='GST')
    st.write(px.line(dff, x='Month', y="GST", color='years', title='Total GST of Each Year'))    
    
    # plot 8
    # import calendar
    df1['Month'] = df1['BillDate'].dt.month
    # To show the total income of Each Month on yearly basis for 2016, 2017 & 2018
    dff = df1.groupby(['Month','years'])['NetAmount'].sum().reset_index(name='NetAmount')
    fig = px.line(dff, x='Month', y="NetAmount", color='years', title='Total Monthly Income of All Regions on Yearly Basis')
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = df1.Month,
        )
    )
    st.write(fig)

    # plot 9    
    # To show the Price for each year i.e., 2016, 2017 & 2018
    dff = df1.groupby(['years'])['Price'].sum().reset_index(name='Price')
    fig = px.bar(dff, x= 'years', y='Price', title='Total Price of Each Year')

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = df1.years,
        )
    )
    fig.update_traces(marker_color='grey')
    st.write(fig)
            
    # plot 10
    # # To show Price of Design for 3 Years i.e., 2016, 2017 & 2018
    dff = df1.groupby(['DesignNo','years'])['Price'].sum().reset_index(name='Price')
    fig = px.bar(dff, x="years", y="Price", color='DesignNo', title='Design Price for 3 Years')
    st.write(fig)
################################################################################
    
with model_training:
    st.header("Prediction for the Cambridge Brand")
    st.subheader('Random Forest Regressor')
# making columns
    input, display=st.columns(2)
# In first column there must be selection point
    max_depth=input.slider("Select max_depth?", min_value=10, max_value=100, value=20, step=10)
# n_estimators
    n_estimators=input.selectbox("n_estimators", options=[50, 100, 200, 300, 'No Limit'], index=0)
# input features from user
input_features= input.selectbox("Select a target input variable to run Random Forest Regressor Model", df.columns)  

# machine learning model
model=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
# if and else condition to justify the no limit
if n_estimators=='No Limit':
    model=RandomForestRegressor(max_depth=max_depth) 
else:
     model=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

# define X and y
x=df[[input_features]]
y=df[['Price']]
model.fit(x, y)
prediction=model.predict(y)

# display matrics
display.subheader('Mean absolute error')
display.write(mean_absolute_error(y, prediction))
display.subheader('Mean squared error')
display.write(mean_squared_error(y, prediction))
display.subheader('r2 score')
display.write(r2_score(y, prediction))
