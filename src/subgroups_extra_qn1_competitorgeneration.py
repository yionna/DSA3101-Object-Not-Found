#We make up these data to train a more complex model that incorporates some external factors:
import pandas as pd
import numpy as np

# Function to generate made-up economic data
def generate_economic_data(num_records=12):

    dates = pd.date_range(start='2023-01-01', periods=num_records, freq='M')

    # Generate random data for economic indicators
    gdp_growth = np.random.uniform(0.5, 3.0, num_records)  # GDP growth (in percentage,same below)
    unemployment_rate = np.random.uniform(3.0, 8.0, num_records)
    consumer_confidence = np.random.uniform(50, 110, num_records)  # Index from 0 to 100+
    interest_rate = np.random.uniform(0.5, 5.0, num_records)
    inflation_rate = np.random.uniform(1.0, 6.0, num_records)
    currency_exchange_rate = np.random.uniform(1.1, 1.5, num_records)  # SGD currency rate against USD

    # Create a DataFrame
    economic_data = pd.DataFrame({
        'Date': dates,
        'GDP_Growth (%)': gdp_growth,
        'Unemployment_Rate (%)': unemployment_rate,
        'Consumer_Confidence_Index': consumer_confidence,
        'Interest_Rate (%)': interest_rate,
        'Inflation_Rate (%)': inflation_rate,
        'Currency_Exchange_Rate (USD)': currency_exchange_rate
    })

    return economic_data

# Function to generate made-up competitor actions data
def generate_competitor_data(num_records=5):
    # Competitor names
    competitors = ['Competitor_A', 'Competitor_B', 'Competitor_C', 'Competitor_D', 'Competitor_E']

    # Actions: pricing strategies, promotions, campaigns, product launches
    pricing_strategy = np.random.choice(['Discount', 'Premium Pricing', 'Bundling'], num_records)
    promotional_offer = np.random.choice(['Buy 1 Get 1', 'Free Shipping', 'Holiday Discount', 'No Offer'], num_records)
    product_launch = np.random.choice([True, False], num_records)
    social_media_engagement = np.random.uniform(1000, 50000, num_records)  # Random social media engagements

    # Create a DataFrame
    competitor_data = pd.DataFrame({
        'Competitor': competitors[:num_records],
        'Pricing_Strategy': pricing_strategy,
        'Promotional_Offer': promotional_offer,
        'Product_Launch': product_launch,
        'Social_Media_Engagement': social_media_engagement
    })

    return competitor_data

# Function to generate made-up industry trends data
def generate_industry_trends(num_records=10):
    # Generate some random trends, industry reports, and disruptions
    trends = ['Sustainability Focus', 'Tech Advancement', 'Regulation Change', 'Supply Chain Disruption', 'Market Consolidation']

    report_dates = pd.date_range(start='2023-01-01', periods=num_records, freq='M')
    trend_type = np.random.choice(trends, num_records)
    impact = np.random.choice(['High', 'Medium', 'Low'], num_records)

    # Create a DataFrame
    industry_trends_data = pd.DataFrame({
        'Report_Date': report_dates,
        'Trend_Type': trend_type,
        'Impact_Level': impact
    })

    return industry_trends_data

# Generate all datasets
economic_data = generate_economic_data()
competitor_data = generate_competitor_data()
industry_trends_data = generate_industry_trends()

import ace_tools as tools; tools.display_dataframe_to_user(name="Economic Data", dataframe=economic_data)
tools.display_dataframe_to_user(name="Competitor Data", dataframe=competitor_data)
tools.display_dataframe_to_user(name="Industry Trends Data", dataframe=industry_trends_data)
