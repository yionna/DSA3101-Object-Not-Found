import pandas as pd
from models.customerSegmentation.static import *
from models.RecommendationSystem.recommendationsystem import obtain_best_columns, model_train_test
import numpy as np
from datetime import datetime, timedelta
from models.EarlyWarningSystem.EarlyWarningSystem import EarlyWarningSystem
from models.CampaignUpdate.UpdateCampaign import CampaignSystem
from models.DynamicCampaign.DynamicCampaignSystem import MainOptimizer
from itertools import product

# Importing data
base_df = pd.read_csv("data/processed/BankChurners_more.csv")
banking_df = pd.read_csv("data/processed/banking_behaviour_preference.csv")
recommendation_df = pd.read_csv("data/processed/recommendation_system_dataset.csv") # dataset with additional product data

# Customer segmentation
segmentation_test = CustomerSegmentation(banking_df)
segmentation_result = segmentation_test.perform_segmentation()

# Recommendation system
clusters = {'low_income': [1, 3], 'medium_income': [2],'high_income': [4, 5]}
product_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
cols, scores = obtain_best_columns(recommendation_df, clusters, product_list)
predicted_labels, actual_labels = model_train_test(recommendation_df, cols, clusters, product_list)

def update_demographic_data(transaction_data, demographic_data):
    """
    Updates the demographic data using groupby and map without using merge.
    
    Parameters:
    - transaction_data: DataFrame containing 'Transaction_Amount(INR)' and 'CLIENTNUM' columns.
    - demographic_data: DataFrame containing 'CLIENTNUM', 'Total_Trans_Amt', 'Total_Trans_Count' columns.
    
    Returns:
    - Updated demographic data DataFrame.
    """
    # Group by Clientnum to calculate the total transaction amount and transaction count
    transaction_sum = transaction_data.groupby('CLIENTNUM')['Transaction_Amount(INR)'].sum()
    transaction_count = transaction_data.groupby('CLIENTNUM')['CLIENTNUM'].count()

    # Update the Total_Trans_Amt in demographic data
    demographic_data['Total_Trans_Amt'] += demographic_data['CLIENTNUM'].map(transaction_sum).fillna(0)
    # Update the Total_Trans_Count in demographic data
    demographic_data['Total_Trans_Count'] += demographic_data['CLIENTNUM'].map(transaction_count).fillna(0)

    return demographic_data

def EWS(demographic_db):
    """
    Runs the Early Warning System (EWS) to identify customers at risk and update the demographic database.
    
    Parameters:
    demographic_db (DataFrame): The current demographic dataset containing customer data.
    
    Returns:
    tuple: A tuple containing:
        - updated_demographic_db (DataFrame): The updated demographic dataset after applying the EWS.
        - high_to_low_risk (DataFrame): Customers who have moved from high to low risk.
        - low_to_high_risk (DataFrame): Customers who have moved from low to high risk.
    """
    # Initialize the early warning system
    ews = EarlyWarningSystem()

    # Apply the early warning system to the current demographic database
    updated_demographic_db = ews.apply_warning(demographic_db)

    # Get the customers who need updates in the campaign database
    high_to_low_risk, low_to_high_risk = ews.get_customers_to_update(demographic_db, updated_demographic_db)

    return updated_demographic_db, high_to_low_risk, low_to_high_risk
# updated_db, high_to_low_risk_customers, low_to_high_risk_customers = EWS(demographic_data)


def get_best_product_per_customer(predicted_labels, df, product_list):
    '''
    Get the best product recommendation for each customer, avoiding already subscribed products.
    Input:
    predicted_labels: dictionary of predicted labels with product probabilities
    df: original dataframe to check product subscription status
    product_list: list of products
    Returns:
    Dictionary of best product recommendation for each customer
    '''
    best_product_per_customer = {}

    for key, products in predicted_labels.items():
        combined_probs = pd.concat(products.values(), axis=1)
        
        for client in combined_probs.index:
            sorted_products = combined_probs.loc[client].sort_values(ascending=False)
            for product in sorted_products.index:
                # Check if the customer has already subscribed to the product
                if df.loc[df['CLIENTNUM'] == client, product].values[0] == 0:
                    if key not in best_product_per_customer:
                        best_product_per_customer[key] = {}
                    best_product_per_customer[key][client] = product
                    break
    
    return best_product_per_customer

#predicted_labels, actual_labels = model_train_test(df, cols, clusters, product_list)
#best_product_per_customer = get_best_product_per_customer(predicted_labels, df, product_list)

def update_campaign_data(campaign_database, best_product_per_customer):
    """
    Main function that updates the campaign database based on the best products per customer.
    
    Parameters:
    campaign_database_path (str): Path to the campaign database CSV file.
    best_product_per_customer (list): List of tuples containing (clientnum, product).
    
    Returns:
    DataFrame: The updated campaign database.
    """
    # Load the existing campaign database
    marketing_system = CampaignSystem(campaign_database)
    
    # Update the campaign database
    updated_campaign_database = marketing_system.update_campaign_database(best_product_per_customer)

    # Return the updated campaign database
    return updated_campaign_database

def calculate_rfm_scores(transaction_data, customer_demographics, financial_status_col='Financial_Status'):
    """
    Calculates RFM (Recency, Frequency, Monetary) scores for each customer based on their transaction data.

    Parameters:
    transaction_data (DataFrame): A DataFrame containing transaction details, including 'CLIENTNUM', 'Transaction_Date', and 'Transaction_Amount'.
    customer_demographics (DataFrame): A DataFrame with customer demographic data, including 'CLIENTNUM' and the specified financial status column.
    financial_status_col (str): The column name in `customer_demographics` representing the financial status of the customer (default is 'Financial_Status').

    Returns:
    Series: A Pandas Series with 'CLIENTNUM' as the index and the calculated RFM score as the value for each customer.
    """
    current_date = datetime.now()
    one_month_ago = current_date - timedelta(days=30)
    
    recent_transactions = transaction_data[transaction_data['Transaction_Date'] >= one_month_ago]
    
    # Recency
    last_transaction_dates = recent_transactions.groupby('CLIENTNUM')['Transaction_Date'].max()
    recency_scores = 1 - ((current_date - last_transaction_dates).dt.days / (current_date - one_month_ago).days)
    
    # Frequency
    transaction_counts = recent_transactions.groupby('CLIENTNUM').size()
    frequency_scores = transaction_counts / transaction_counts.max()
    
    # Monetary
    total_spending = recent_transactions.groupby('CLIENTNUM')['Transaction_Amount'].sum()
    max_spending = total_spending.max()
    financial_status_scores = customer_demographics.set_index('CLIENTNUM')[financial_status_col]
    monetary_scores = (total_spending / max_spending) * financial_status_scores
    monetary_scores = monetary_scores.fillna(0) / monetary_scores.max()

    # Weighted RFM score
    rfm_scores = 0.3 * recency_scores.fillna(0) + 0.3 * frequency_scores.fillna(0) + 0.4 * monetary_scores

    return rfm_scores

# Main function to update the campaign log with selected strategies
def update_campaign_log(campaign_log, channels, timing, frequency, transaction_data, customer_demographics):
    """
    Updates the campaign log by selecting and applying the best strategy for each customer based on chosen arms.

    Parameters:
    campaign_log (DataFrame): The campaign log containing data about current and past campaigns.
    channels (list): List of available channels for the marketing strategy (e.g., 'email', 'SMS').
    timing (list): List of available timings for campaign delivery (e.g., 12, 18 ).
    frequency (list): List of available campaign frequencies (e.g., 1,2,3, weekly base).
    optimizer (object): An optimization model that selects the best arm (combination of channel, timing, and frequency) based on customer features.
    transaction_data (DataFrame): A DataFrame containing transaction data for calculating RFM scores.
    customer_demographics (DataFrame): A DataFrame containing customer demographic information.

    Returns:
    DataFrame: The updated campaign log with the chosen strategies for each customer.
    """
    optimizer = MainOptimizer()
    # Create all combinations of channel, timing, and frequency as possible arms
    arms = list(product(channels, timing, frequency))
    arm_mapping = {index: arm for index, arm in enumerate(arms)}

    # Iterate over the campaign log and update with the chosen strategy
    for index, row in campaign_log.iterrows():
        # Simulate or extract customer features (using random features for demo)
        customer_features = ['Customer_Age', 'Gender', 'Num_of_Contacts_Made', 'Financial_Status', 'Total_Trans_Count',
                       'Savings', 'Total_Trans_Amt', 'Balance', 'Credit_Score', 'Outstanding_Loans']
        
        # Select the best arm for the customer
        chosen_arm_index = optimizer.select_arm(customer_features)
        chosen_arm = arm_mapping[chosen_arm_index]

        # Simulate feedback (using existing row data)
        conversion_rate = row['conversion_rate']
        clickthrough_rate = row['clickthroughrate']
        rfm_score = calculate_rfm_scores(transaction_data, customer_demographics)

        # Update the campaign log with the chosen strategy
        campaign_log.at[index, 'ChosenChannel'] = chosen_arm[0]
        campaign_log.at[index, 'ChosenTiming'] = chosen_arm[1]
        campaign_log.at[index, 'ChosenFrequency'] = chosen_arm[2]

        # Update the optimizer based on the feedback
        optimizer.update(chosen_arm_index, customer_features, conversion_rate, clickthrough_rate, rfm_score)
    
    return campaign_log
