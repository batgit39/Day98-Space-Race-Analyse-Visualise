#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <center><img src="https://i.imgur.com/9hLRsjZ.jpg" height=400></center>
# 
# This dataset was scraped from [nextspaceflight.com](https://nextspaceflight.com/launches/past/?page=1) and includes all the space missions since the beginning of Space Race between the USA and the Soviet Union in 1957!

# ### Install Package with Country Codes

# In[1]:


get_ipython().run_line_magic('pip', 'install iso3166')


# ### Upgrade Plotly
# 
# Run the cell below if you are working with Google Colab.

# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade plotly')


# ### Import Statements

# In[3]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# These might be helpful:
from iso3166 import countries
from datetime import datetime, timedelta


# ### Notebook Presentation

# In[4]:


pd.options.display.float_format = '{:,.2f}'.format


# ### Load the Data

# In[5]:


df_data = pd.read_csv('mission_launches.csv')


# # Preliminary Data Exploration
# 
# * What is the shape of `df_data`? 
# * How many rows and columns does it have?
# * What are the column names?
# * Are there any NaN values or duplicates?

# In[6]:


df_data.head()


# In[7]:


df_data.shape


# In[8]:


df_data.columns


# In[9]:


df_data.isna().values.any()


# ## Data Cleaning - Check for Missing Values and Duplicates
# 
# Consider removing columns containing junk data. 

# In[10]:


columns_to_drop = ['Unnamed: 0.1','Unnamed: 0']
df = df_data.drop(columns_to_drop, axis=1)


# In[11]:


df.head()


# ## Descriptive Statistics

# In[12]:


df.describe()


# # Number of Launches per Company
# 
# Create a chart that shows the number of space mission launches by organisation.

# In[13]:


launch_count = df.Organisation.value_counts()
launch_count = launch_count.sort_values(ascending=False)


# In[14]:


plt.figure(figsize=(20, 12)) 
plt.barh(launch_count.index, launch_count.values)
plt.ylabel('Organisation', fontsize=5)
plt.xlabel('Number of Space Mission Launches')
plt.title('Number of Space Mission Launches by Organisation')
plt.show()


# # Number of Active versus Retired Rockets
# 
# How many rockets are active compared to those that are decomissioned? 

# In[15]:


active_rockets = df[df['Rocket_Status'] == 'StatusActive'].shape[0]
active_rockets = df['Rocket_Status'].value_counts()


# In[16]:


plt.barh(active_rockets.index, active_rockets.values)
plt.ylabel('Rocket Status', fontsize=12)
plt.xlabel('Number of Rockets')
plt.title('Number of Active versus Retired Rockets')
plt.show()


# # Distribution of Mission Status
# 
# How many missions were successful?
# How many missions failed?

# In[17]:


mission_status = df[df['Mission_Status'] == 'Success'].shape[0]
mission_status = df['Mission_Status'].value_counts()


# In[18]:


plt.barh(mission_status.index, mission_status.values)
plt.ylabel('Mission Status', fontsize=12)
plt.xlabel('Number of Missions')
plt.title('Distribution of Mission Status')
plt.show()


# # How Expensive are the Launches? 
# 
# Create a histogram and visualise the distribution. The price column is given in USD millions (careful of missing values). 

# In[19]:


df_clean = df.dropna(subset=['Price'])
df_clean.Price.isna().values.any()


# In[20]:


plt.figure(figsize=(8, 6))
plt.hist(df_clean["Price"], bins=10, edgecolor='black')

plt.xlabel('Price (USD millions)')
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# # Use a Choropleth Map to Show the Number of Launches by Country
# 
# * Create a choropleth map using [the plotly documentation](https://plotly.com/python/choropleth-maps/)
# * Experiment with [plotly's available colours](https://plotly.com/python/builtin-colorscales/). I quite like the sequential colour `matter` on this map. 
# * You'll need to extract a `country` feature as well as change the country names that no longer exist.
# 
# Wrangle the Country Names
# 
# You'll need to use a 3 letter country code for each country. You might have to change some country names.
# 
# * Russia is the Russian Federation
# * New Mexico should be USA
# * Yellow Sea refers to China
# * Shahrud Missile Test Site should be Iran
# * Pacific Missile Range Facility should be USA
# * Barents Sea should be Russian Federation
# * Gran Canaria should be USA
# 
# 
# You can use the iso3166 package to convert the country names to Alpha3 format.

# # Use a Choropleth Map to Show the Number of Failures by Country
# 

# In[21]:


df['Country'] = df['Location'].str.split(',').str[-1].str.strip()
df_sorted = df.sort_values('Country')


# In[22]:


fig = px.choropleth(df_sorted, locations='Country', locationmode='country names', color='Mission_Status',
                    projection='natural earth')
fig.update_layout(title_text='Choropleth Map', title_x=0.5)


# # Create a Plotly Sunburst Chart of the countries, organisations, and mission status. 

# In[23]:


fig = px.sunburst(df, path=['Location', 'Organisation', 'Mission_Status'])
fig.update_layout(title_text='Sunburst Chart - Countries, Organisations, and Mission Status')
fig.show()


# # Analyse the Total Amount of Money Spent by Organisation on Space Missions

# In[24]:


total_spending = df.groupby('Organisation')['Price'].sum()
print(total_spending)


# # Analyse the Amount of Money Spent by Organisation per Launch

# In[25]:


money_spent = df_data[df_data["Price"].notna()].copy()
money_spent["Price"] = money_spent["Price"].str.replace(',', '').astype(float)


# In[26]:


organisation_expense = money_spent.groupby("Organisation")["Price"].mean().reset_index()
organisation_expense.sort_values("Price", ascending=False)
organisation_expense.head()


# # Chart the Number of Launches per Year

# In[95]:


import datetime

def extract_year(date_str):
    try:
        dt = datetime.datetime.strptime(date_str, "%a %b %d, %Y %H:%M %Z")
    except ValueError:
        dt = datetime.datetime.strptime(date_str, "%a %b %d, %Y")
    return dt.year

df['Year'] = df['Date'].apply(extract_year)


# In[96]:


ds = df['Year'].value_counts().reset_index()
ds.columns = [
    'Year', 
    'Count'
]
fig = px.bar(
    ds, 
    x='Year', 
    y="Count", 
    orientation='v', 
    title='Number Of launches Per Year' 
)
fig.show()


# # Chart the Number of Launches Month-on-Month until the Present
# 
# Which month has seen the highest number of launches in all time? Superimpose a rolling average on the month on month time series chart. 

# In[128]:


def extract_year(date_str):
    try:
        dt = datetime.datetime.strptime(date_str, "%a %b %d, %Y %H:%M %Z")
    except ValueError:
        dt = datetime.datetime.strptime(date_str, "%a %b %d, %Y")
    return dt.month

df['Month'] = df['Date'].apply(extract_year)


# In[98]:


month_on_month = df['Month'].value_counts().reset_index()
month_on_month.columns = [
    'Month', 
    'Count'
]
fig = px.bar(
    month_on_month, 
    x='Month', 
    y="Count", 
    orientation='v', 
    title='Sum of total missions in each Month',
    color='Count'
)
fig.show()


# # Launches per Month: Which months are most popular and least popular for launches?
# 
# Some months have better weather than others. Which time of year seems to be best for space missions?

# In[99]:


month_on_month.max()


# In[100]:


month_on_month.min()


# # How has the Launch Price varied Over Time? 
# 
# Create a line chart that shows the average price of rocket launches over time. 

# In[101]:


df_price = df.sort_values(by='Price', ascending=False)


# In[102]:


fig = px.line(df_price, x='Price', y='Year', title='Sales Trend')
fig.show()


# In[106]:


avg_price = df[df["Price"].notna()]
pd.options.mode.chained_assignment = None
avg_price["Price"] = avg_price["Price"].str.replace(',', '').astype(float)

avg_price.drop(columns=['Detail','year', 'Location', 'Organisation', 'Country','Date','Rocket_Status','Mission_Status'], inplace=True)
avg_price.head()


# In[108]:


avg_price.groupby("Year").mean().plot(figsize=(12, 8))


# In[73]:


groups = avg_price.groupby("Year")
fig, ax = plt.subplots(figsize=(12, 8))
for year, group in groups:
    group.plot(x="Date", y="Price", ax=ax, label=year)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Average Price by Year")
ax.legend()
plt.show()


# # Chart the Number of Launches over Time by the Top 10 Organisations. 
# 
# How has the dominance of launches changed over time between the different players? 

# In[74]:


df.head()


# In[70]:


groups = avg_price.groupby("Year")
fig, ax = plt.subplots(figsize=(12, 8))

# Initialize the color palette
colors = plt.cm.tab10.colors

for i, (year, group) in enumerate(groups):
    # Set the alpha value based on whether the year is selected or not
    alpha = 1.0 if i == 0 else 0.3

    # Plot the data for the year with the specified color and alpha value
    group.plot(x="Date", y="Price", ax=ax, label=year, color=colors[i%10], alpha=alpha)

# Set labels and title
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Average Price by Year")

# Show the legend
ax.legend()

# Function to update the alpha values on click
def onclick(event):
    # Get the index of the clicked line
    index = event.ind[0]

    # Update the alpha values of all lines
    for i, line in enumerate(ax.lines):
        line.set_alpha(1.0 if i == index else 0.3)

    # Redraw the figure
    fig.canvas.draw()

# Connect the onclick event to the figure
fig.canvas.mpl_connect('pick_event', onclick)

# Display the plot
plt.show()


# In[120]:


top_10_org = pd.DataFrame(columns=df.columns)

for val in df.groupby("Organisation").count().sort_values("Date",ascending=False)[:10].index:
    print(val)
    org = df[df.Organisation == val]
    top_10_org = pd.concat([top_10, org], ignore_index=True)

top_10_org


# In[160]:


def extract_year(date_str):
    try:
        dt = datetime.datetime.strptime(date_str, "%a %b %d, %Y %H:%M %Z")
    except ValueError:
        dt = datetime.datetime.strptime(date_str, "%a %b %d, %Y")
    return dt.decade


# In[161]:


df[df.Organisation=="CASC"]
top_10_org.groupby("Organisation").count().sort_values("Date",ascending=False)[:10].index
px.histogram(top_10_org.sort_values(by=["Organisation", "Date"], ascending=[True, False]),
             x="Organisation",
             color='Organisation',
             nbins=10) 


# In[ ]:





# # Cold War Space Race: USA vs USSR
# 
# The cold war lasted from the start of the dataset up until 1991. 

# In[189]:


df['Country'].unique()
countries_to_replace = ['Kazakhstan', 'Russia']
df.loc[df['Country'].isin(countries_to_replace), 'Country'] = 'Russia'


# In[190]:


CW_df = df[(df['Country']=='USA') | (df['Country']=='Russia')]


# In[191]:


war = CW_df.sort_values("Year")
war[(war.Year <= 1991)]


# ## Create a Plotly Pie Chart comparing the total number of launches of the USSR and the USA
# 
# Hint: Remember to include former Soviet Republics like Kazakhstan when analysing the total number of launches. 

# In[192]:


CW_df["Country"].value_counts().rename_axis("Country").reset_index(name='counts')


# In[199]:


colors = ["purple", "orange"]
grouping = CW_df.groupby("Country").count().reset_index()
sizes = grouping['Mission_Status']
labels = grouping['Country']

plt.pie(sizes, labels = labels, colors = colors)


# ## Create a Chart that Shows the Total Number of Launches Year-On-Year by the Two Superpowers

# In[202]:


CW_df.groupby(["year", "Country"]).size().unstack().plot()


# ## Chart the Total Number of Mission Failures Year on Year.

# In[206]:


mission_failures = CW_df[CW_df['Mission_Status'] == 'Failure']
failures_yearly = mission_failures.groupby('Year').size()

plt.figure(figsize=(12, 8))
failures_yearly.plot(kind='bar', color='red')
plt.xlabel('Year')
plt.ylabel('Number of Mission Failures')
plt.title('Total Number of Mission Failures Year on Year (Bar Plot)')
plt.show()

plt.figure(figsize=(12, 8))
failures_yearly.plot(kind='line', marker='o', color='blue')
plt.xlabel('Year')
plt.ylabel('Number of Mission Failures')
plt.title('Total Number of Mission Failures Year on Year (Line Plot)')
plt.grid(True)
plt.show()


# ## Chart the Percentage of Failures over Time
# 
# Did failures go up or down over time? Did the countries get better at minimising risk and improving their chances of success over time? 

# In[210]:


total_missions = CW_df.groupby('year').size()
failures = CW_df[CW_df['Mission_Status'] == 'Failure'].groupby('year').size()
failure_percentage = (failures / total_missions) * 100

failure_percentage.plot(kind='line', figsize=(10, 6))
plt.title('Percentage of Failures over Time')
plt.xlabel('Year')
plt.ylabel('Failure Percentage')
plt.show()


# # For Every Year Show which Country was in the Lead in terms of Total Number of Launches up to and including including 2020)
# 
# Do the results change if we only look at the number of successful launches? 

# In[217]:


df.Country.nunique()


# In[218]:


df.head()


# In[224]:


top_countries = []

for year in df['Year'].unique():
    year_data = df[df['Year'] == year]
    top_country = year_data['Country'].value_counts().idxmax()
    top_countries.append((year, top_country))
    
top_countries_df = pd.DataFrame(top_countries, columns=['Year', 'Top Country'])

plt.figure(figsize=(10, 6))
plt.bar(top_countries_df['Year'], top_countries_df['Top Country'])
plt.xlabel('Year')
plt.ylabel('Top Country')
plt.title('Top Country with Most Launches Each Year')

plt.xticks(rotation=90)
plt.show()


# # Create a Year-on-Year Chart Showing the Organisation Doing the Most Number of Launches
# 
# Which organisation was dominant in the 1970s and 1980s? Which organisation was dominant in 2018, 2019 and 2020? 

# In[249]:


org_launches = df.groupby("year")["Organisation"].value_counts().rename_axis(["year", "Organisation"]).reset_index(name='counts')
org_launches.loc[org_launches.groupby("year")["counts"].idxmax()]
org_launches.head()


# In[251]:


org_set = set(org_launches['Organisation'])

plt.figure(figsize=(12, 10), dpi=80)
for org in org_set:
    selected_data = org_launches.loc[org_launches['Organisation'] == org]
    plt.plot(selected_data['year'], selected_data['counts'], label=org)
     
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=6)
plt.show()


# In[ ]:




