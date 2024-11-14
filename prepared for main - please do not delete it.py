import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from datetime import datetime, timedelta
from models.EarlyWarningSystem.EarlyWarningSystem import EarlyWarningSystem
from models.CampaignUpdate.UpdateCampaign import CampaignSystem
from models.DynamicCampaign.DynamicCampaignSystem import MainOptimizer
from itertools import product

def update_demographic_data(transaction_data, demographic_data):
    """
    Updates the demographic data using groupby and map without using merge.
    
    Parameters:
    - transaction_data: DataFrame containing 'Transaction_Amount(INR)' and 'Clientnum' columns.
    - demographic_data: DataFrame containing 'Clientnum', 'Total_Trans_Amt', 'Total_Trans_Count' columns.
    
    Returns:
    - Updated demographic data DataFrame.
    """
    # Group by Clientnum to calculate the total transaction amount and transaction count
    transaction_sum = transaction_data.groupby('Clientnum')['Transaction_Amount(INR)'].sum()
    transaction_count = transaction_data.groupby('Clientnum')['Clientnum'].count()

    # Update the Total_Trans_Amt in demographic data
    demographic_data['Total_Trans_Amt'] += demographic_data['Clientnum'].map(transaction_sum).fillna(0)
    # Update the Total_Trans_Count in demographic data
    demographic_data['Total_Trans_Count'] += demographic_data['Clientnum'].map(transaction_count).fillna(0)

    return demographic_data

def EWS(demographic_db):
    # Initialize the early warning system
    ews = EarlyWarningSystem()

    # Apply the early warning system to the current demographic database
    updated_demographic_db = ews.apply_warning(demographic_db)

    # Get the customers who need updates in the campaign database
    high_to_low_risk, low_to_high_risk = ews.get_customers_to_update(demographic_db, updated_demographic_db)

    return updated_demographic_db, high_to_low_risk, low_to_high_risk
#updated_db, high_to_low_risk_customers, low_to_high_risk_customers = EWS(demographic_data)

def model_train_test(df, cols, product_list, clusters):
    training_features = {key: {} for key in clusters.keys()}
    predicted_labels = {key: {} for key in clusters.keys()}
    actual_labels = {key: {} for key in clusters.keys()}

    # Loop through clusters and products
    for key, cluster in clusters.items():
        users = df[df['Segment'].isin(cluster)]
        for i in product_list:
            # Define the target variable for the product
            y = users[i]
            
            # Set up training features from selected columns
            training_features[key][i] = users[list(cols[key][i]) + ['CLIENTNUM']]
            training_features[key][i] = training_features[key][i].set_index('CLIENTNUM')
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(training_features[key][i], y, test_size=0.4, random_state=123)
            
            # Train XGBClassifier
            bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
            bst.fit(X_train.to_numpy(), y_train.to_numpy())
            
            # Prediction
            predicted_labels[key][i] = pd.DataFrame(bst.predict_proba(X_test.to_numpy())[:, 1], index=X_test.index, columns=[i])
            actual_labels[key][i] = pd.DataFrame(y_test, index=X_test.index, columns=['Actual'])
    return training_features, predicted_labels, actual_labels

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
    current_date = datetime.now()
    one_month_ago = current_date - timedelta(days=30)
    
    recent_transactions = transaction_data[transaction_data['Transaction_Date'] >= one_month_ago]
    
    #Recency
    last_transaction_dates = recent_transactions.groupby('CLIENTNUM')['Transaction_Date'].max()
    recency_scores = 1 - ((current_date - last_transaction_dates).dt.days / (current_date - one_month_ago).days)
    #Frequency
    transaction_counts = recent_transactions.groupby('CLIENTNUM').size()
    frequency_scores = transaction_counts / transaction_counts.max()
    #Monetary
    total_spending = recent_transactions.groupby('CLIENTNUM')['Transaction_Amount'].sum()
    max_spending = total_spending.max()
    financial_status_scores = customer_demographics.set_index('CLIENTNUM')[financial_status_col]
    monetary_scores = (total_spending / max_spending) * financial_status_scores

    monetary_scores = monetary_scores.fillna(0) / monetary_scores.max()

    rfm_scores = 0.3 * recency_scores.fillna(0) + 0.3 * frequency_scores.fillna(0) + 0.4 * monetary_scores

    return rfm_scores

# Main function to update the campaign log with selected strategies
def update_campaign_log(campaign_log, channels, timing, frequency, optimizer, transaction_data, customer_demographics):
    # Create all combinations of channel, timing, and frequency as possible arms
    arms = list(product(channels, timing, frequency))
    arm_mapping = {index: arm for index, arm in enumerate(arms)}

    # Iterate over the campaign log and update with the chosen strategy
    for index, row in campaign_log.iterrows():
        # Simulate or extract customer features (using random features for demo)
        customer_features = np.random.rand(10)  # Replace with actual features if available
        
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