import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Simulate generating multiple sets of customer campaign data
np.random.seed(3101)
n_customers = 1000
n_iterations = 5  # Define the number of data sets

# Function to generate simulated campaign data
def generate_campaign_data(n_customers):
    return pd.DataFrame({
        'CLIENTNUM': np.random.choice(np.arange(1, n_customers + 1), n_customers * 5),
        'Campaign_ID': np.random.randint(1000, 5000, n_customers * 5),
        'Campaign_Type': np.random.choice(['Retention', 'Conversion', 'Consideration'], n_customers * 5),
        'EngagementStyle': np.random.choice(['Formal', 'Friendly', 'Humorous', 'Urgency', 'Inspirational', 
                                             'Technical', 'Emotional', 'Direct', 'Luxury', 'Educational'], n_customers * 5),
        'Conversion_Rate': np.random.uniform(0, 1, n_customers * 5)  # Random conversion rate between 0 and 1
    })

# Create multiple sets of campaign data
campaign_data_sets = [generate_campaign_data(n_customers) for _ in range(n_iterations)]

# Define layout for the dashboard
app.layout = html.Div([
    html.H1("Customer Preference for Marketing Styles Dashboard"),
    dcc.Graph(id='pie-chart'),
    dcc.Graph(id='bar-chart'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every 5 seconds
        n_intervals=0
    )
])

# Define callback to update the charts
@app.callback(
    [Output('pie-chart', 'figure'),
     Output('bar-chart', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_charts(n_intervals):
    # Select the current data set based on the interval count
    current_data = campaign_data_sets[n_intervals % len(campaign_data_sets)]

    # Calculate the most effective style for each customer based on average conversion rate
    style_summary = current_data.groupby(['CLIENTNUM', 'EngagementStyle'])['Conversion_Rate'].mean().reset_index()
    preferred_styles = style_summary.loc[style_summary.groupby('CLIENTNUM')['Conversion_Rate'].idxmax()]

    # Count the number of customers preferring each style
    style_counts = preferred_styles['EngagementStyle'].value_counts()

    # Create a pie chart
    pie_fig = px.pie(
        values=style_counts.values,
        names=style_counts.index,
        title='Customer Preference for Marketing Styles Based on Conversion Rates',
        hole=0.3  # For a donut-style chart, set this value between 0 and 1
    )

    # Create a bar chart
    bar_fig = px.bar(
        x=style_counts.index,
        y=style_counts.values,
        labels={'x': 'Engagement Style', 'y': 'Number of Customers'},
        title='Number of Customers Preferring Each Engagement Style',
        text_auto=True
    )

    return pie_fig, bar_fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

