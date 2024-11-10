from datetime import datetime
from datetime import timedelta
import uuid
import numpy as np

# this is a function to extract the financial status and loyalty status from the customer label
def extract_financial_and_loyalty_status(customer_id, customer_df):

    customer_info = customer_df[customer_df['CustomerID'] == customer_id]
    if not customer_info.empty:
        segmentation = customer_info['Segmentation'].values[0]
        financial_status = segmentation.split(',')[0].strip()
        loyalty_level = segmentation.split(',')[1].strip()
        
        # Extract only the status values (e.g., 'High', 'Moderate', 'Low')
        financial_status = financial_status.replace('Financial status', '').strip()
        loyalty_level = loyalty_level.replace('loyalty_level', '').strip()
        return financial_status, loyalty_level
    else:
        return None
class EventTriggerSystem:
    def __init__(self, transaction_data, app_interaction_data, campaign_database):
        self.transaction_data = transaction_data
        self.app_interaction_data = app_interaction_data
        self.campaign_database = campaign_database
        self.active_campaigns = {}

        self.campaign_durations = {
            'Cashback Promotion': timedelta(weeks=4),
            'Credit Card Promotion': timedelta(weeks=8),
            'Loan Promotion': timedelta(weeks=12),
            'Insurance Promotion': timedelta(weeks=24),
            'Investment Product Promotion': timedelta(weeks=8)
        }

    def generate_campaign_id(self, campaign_type):
        """
        Generate a new Campaign ID and set its validity period.
        """
        campaign_id = str(uuid.uuid4())
        expiration_date = datetime.now() + self.campaign_durations.get(campaign_type, timedelta(weeks=4))
        self.active_campaigns[campaign_type] = {
            'CampaignID': campaign_id,
            'ExpirationDate': expiration_date,
            'StartDate': datetime.now(),
            'EndDate': expiration_date
        }
        return campaign_id

    def update_campaign_database(self, client, campaign_type):
        """
        Update the Campaign Database with triggered Campaign details.
        """
        current_time = datetime.now()
        campaign_info = self.active_campaigns.get(campaign_type)

        if not campaign_info or current_time > campaign_info['ExpirationDate']:
            campaign_id = self.generate_campaign_id(campaign_type)
            start_date = datetime.now()
            end_date = self.active_campaigns[campaign_type]['EndDate']
        else:
            campaign_id = campaign_info['CampaignID']
            start_date = campaign_info['StartDate']
            end_date = campaign_info['EndDate']

        self.campaign_database.append({
            'CampaignID': campaign_id,
            'Campaign_Type': campaign_type,
            'CLIENTNUM': client,
            'Start_Date': start_date,
            'End_Date': end_date,
            'Response_Status': 'unknown' 
        })

    def trigger_campaign(self, client, campaign_type, condition):
        """
        Trigger a campaign based on a given condition and update the campaign database.
        """
        if condition(client):
            self.update_campaign_database(client, campaign_type)
            return True
        return False

    def trigger_cashback_promotion(self, client):
        condition = lambda client: max(
            [txn['Transaction_Date'] for txn in self.transaction_data if txn['CLIENTNUM'] == client], 
            default=None
        ) and (datetime.now() - max(
            [txn['Transaction_Date'] for txn in self.transaction_data if txn['CLIENTNUM'] == client]
        )).days >= 14
        return self.trigger_campaign(client, 'Cashback Promotion', condition)

    def trigger_credit_card_promotion(self, client):
        condition = lambda client: all(
            not txn.get('Default', False) for txn in self.transaction_data if txn['CLIENTNUM'] == client and txn['Transaction_Date'] >= datetime.now() - timedelta(weeks=26)
        ) or any(
            txn['Transaction_Amount'] > 1000 for txn in self.transaction_data if txn['CLIENTNUM'] == client
        )
        return self.trigger_campaign(client, 'Credit Card Promotion', condition)

    def trigger_loan_promotion(self, client):
        condition = lambda client: any(
            txn['Transaction_Type'] in ['Housing', 'Car'] or txn['Transaction_Amount'] > 10000 for txn in self.transaction_data if txn['CLIENTNUM'] == client
        ) or any(
            interaction['CLIENTNUM'] == client and interaction['Page_Type'] in ['Housing Loan', 'Car Loan', 'Business Loan'] and interaction['Time_Spent'] > 30 for interaction in self.app_interaction_data
        )
        return self.trigger_campaign(client, 'Loan Promotion', condition)

    def trigger_insurance_promotion(self, client):
        condition = lambda client: any(
            txn['Transaction_Type'] in ['Medical', 'Housing'] or txn['Transaction_Amount'] > 10000 for txn in self.transaction_data if txn['CLIENTNUM'] == client and txn['Transaction_Date'] >= datetime.now() - timedelta(weeks=4)
        ) or any(
            txn['Transaction_Type'] in ['Housing Loan', 'Car Loan'] for txn in self.transaction_data if txn['CLIENTNUM'] == client
        )
        return self.trigger_campaign(client, 'Insurance Promotion', condition)

    def trigger_investment_product_promotion(self, client):
        condition = lambda client: any(
            interaction['CLIENTNUM'] == client and interaction['Page_Type'] == 'Investment' and interaction['Time_Spent'] > 30 for interaction in self.app_interaction_data
        )
        return self.trigger_campaign(client, 'Investment Product Promotion', condition)

def adjust_threshold(client, current_threshold, financial_status):
    if financial_status == 'High Financial Status':
        new_threshold = max(current_threshold - 500, 2500)
    elif financial_status in ['Low Financial Status', 'Moderate Financial Status']:
        new_threshold = max(current_threshold - 200, 1000)
    else:
        new_threshold = current_threshold

    print(f"Adjusted cashback threshold for Client {client}: New Threshold = {new_threshold}")
    return new_threshold

def adjust_amount(client, current_amount, financial_status):
    if financial_status == 'High Financial Status':
        new_amount = min(current_amount + 50, current_amount * 1.07)
    elif financial_status in ['Low Financial Status', 'Moderate Financial Status']:
        new_amount = min(current_amount + 20, current_amount * 1.05)
    else:
        new_amount = current_amount

    print(f"Adjusted cashback amount for Client {client}: New Amount = {new_amount}")
    return new_amount


def increase_limit(client, current_limit, financial_status):
    if financial_status == 'High Financial Status':
        new_limit = min(current_limit * 1.3, current_limit + 1500)
    elif financial_status == 'Moderate Financial Status':
        new_limit = min(current_limit * 1.2, current_limit + 1000)
    elif financial_status == 'Low Financial Status':
        new_limit = min(current_limit * 1.1, current_limit + 500)
    else:
        new_limit = current_limit

    print(f"Increased credit limit for Client {client}: New Limit = {new_limit}")
    return new_limit

def reduce_interest(client, current_interest_rate, financial_status):
    if financial_status == 'High Financial Status':
        new_interest_rate = max(current_interest_rate - 1.5, 3)
    elif financial_status == 'Moderate Financial Status':
        new_interest_rate = max(current_interest_rate - 1, 3)
    else:
        new_interest_rate = max(current_interest_rate - 0.5, 3)

    print(f"Reduced interest rate for Client {client}: New Rate = {new_interest_rate}%")
    return new_interest_rate

def reduce_interest_loan(client, current_interest_rate, financial_status):
    if financial_status == 'High Financial Status':
        new_interest_rate = max(current_interest_rate - 1.5, 3)
    elif financial_status in ['Low Financial Status', 'Moderate Financial Status']:
        new_interest_rate = max(current_interest_rate - 1, 3)
    else:
        new_interest_rate = current_interest_rate

    print(f"Reduced loan interest rate for Client {client}: New Rate = {new_interest_rate}%")
    return new_interest_rate

def long_term_loan(client, current_loan_term):
    new_loan_term = current_loan_term + 12  # Extend by 12 months by default
    print(f"Extended loan term for Client {client}: New Term = {new_loan_term} months")
    return new_loan_term

def loyalty_bonus(client, loyalty_cat):
    if loyalty_cat == "High Loyalty":
        print(f"Loyalty bonus offered for Client {client}")
        return "Loyalty bonus offered"

def price_discount(client, current_price):
    new_price = max(current_price * 0.85, current_price - 100)  # Ensure price discount is not lower than 85% of the original price
    print(f"Price discounted for Client {client}: New Price = {new_price}")
    return new_price

def increasing_insurance(client, current_coverage):
    new_coverage = min(current_coverage * 1.15, current_coverage + 5000)  # Ensure insurance amount not higher than 115% of original
    print(f"Increased insurance coverage for Client {client}: New Coverage = {new_coverage}")
    return new_coverage

def price_discount(client, current_price):
    new_price = max(current_price * 0.85, current_price - 200)  
    print(f"Price discounted for Client {client}: New Price = {new_price}")
    return new_price

def increasing_interest(client, current_interest_rate, financial_status):
    if financial_status == 'High Financial Status':
        new_interest_rate = min(current_interest_rate + 1.5, 7)
    else:
        new_interest_rate = min(current_interest_rate + 1, 7)

    print(f"Increased interest rate for Client {client}: New Rate = {new_interest_rate}%")
    return new_interest_rate

def early_withdrawal_bonus(client):
    print(f"Early withdrawal bonus offered for Client {client}")
    return "Early withdrawal bonus activated"
class GradientBanditOptimizer:
    def __init__(self, learning_rate=0.1):
        # Define actions specific to each campaign type
        self.campaign_actions = {
            'Cashback Promotion': ["adjust_threshold", "adjust_return_amount", "add_limited_offer"],
            'Credit Card Promotion': ["increase_limit", "reduce_interest"],
            'Loan Promotion': ["reduce_interest_loan", "long_term_loan", "loyalty_bonus"],
            'Insurance Promotion': ["price_discount", "increasing_insurance", "limited_add_on_coverage"],
            'Investment Product Promotion': ["price_discount", "increasing_interest", "early_withdrawal_bonus"]
        }
        self.learning_rate = learning_rate
        self.baseline = 0.0
        self.preferences = {}  # Store preferences for each campaign type

        # Initialize preferences for each campaign type's actions
        for campaign_type, actions in self.campaign_actions.items():
            self.preferences[campaign_type] = np.zeros(len(actions))

    def select_action(self, campaign_type):
        """
        Select an action based on current preferences using softmax probability for a specific campaign type.
        """
        exp_preferences = np.exp(self.preferences[campaign_type] - np.max(self.preferences[campaign_type]))
        action_probabilities = exp_preferences / exp_preferences.sum()
        action_index = np.random.choice(len(self.campaign_actions[campaign_type]), p=action_probabilities)
        selected_action = self.campaign_actions[campaign_type][action_index]
        return action_index, selected_action

    def act_on_action(self, client, action, current_params, financial_status=None):
        """
        Execute the selected action and modify campaign parameters.
        """
        # Define the logic to act based on the action (functions need to be defined as appropriate)
        if action == "adjust_threshold":
            return adjust_threshold(client, current_params['threshold'], financial_status)
        elif action == "adjust_return_amount":
            return adjust_amount(client, current_params['amount'], financial_status)
        elif action == "increase_limit":
            return increase_limit(client, current_params['limit'], financial_status)
        elif action == "reduce_interest":
            return reduce_interest(client, current_params['interest_rate'], financial_status)
        elif action == "reduce_interest_loan":
            return reduce_interest_loan(client, current_params['interest_rate'], financial_status)
        elif action == "long_term_loan":
            return long_term_loan(client, current_params['loan_term'])
        elif action == "loyalty_bonus":
            return loyalty_bonus(client, current_params['loyalty_cat'])
        elif action == "price_discount":
            return price_discount(client, current_params['price'])
        elif action == "increasing_insurance":
            return increasing_insurance(client, current_params['coverage'])
        elif action == "increasing_interest":
            return increasing_interest(client, current_params['interest_rate'], financial_status)
        elif action == "early_withdrawal_bonus":
            return early_withdrawal_bonus(client)
        else:
            print(f"No valid action found for {action}")
            return None

    def update_preferences(self, campaign_type, action_index, reward):
        """
        Update preferences based on the observed reward (conversion rate) for a specific campaign type.
        """
        self.baseline += self.learning_rate * (reward - self.baseline)
        exp_preferences = np.exp(self.preferences[campaign_type] - np.max(self.preferences[campaign_type]))
        action_probabilities = exp_preferences / exp_preferences.sum()

        for i in range(len(self.preferences[campaign_type])):
            if i == action_index:
                self.preferences[campaign_type][i] += self.learning_rate * (reward - self.baseline) * (1 - action_probabilities[i])
            else:
                self.preferences[campaign_type][i] -= self.learning_rate * (reward - self.baseline) * action_probabilities[i]

def cashback_promotion_static(financial_status):
    if financial_status == 'High Financial Status':
        return {"threshold": 4000, "amount": 200}  # $200 cashback for $4000 spend
    elif financial_status == 'Moderate Financial Status':
        return {"threshold": 3000, "amount": 150}  # $150 cashback for $3000 spend
    elif financial_status == 'Low Financial Status':
        return {"threshold": 1500, "amount": 100}  # $100 cashback for $1500 spend
    else:
        return {}

def credit_card_promotion_static(financial_status):
    if financial_status == 'High Financial Status':
        return {"credit_limit_increase": 10000}  # $10,000 credit limit increase
    elif financial_status == 'Moderate Financial Status':
        return {"credit_limit_increase": 5000}  # $5,000 credit limit increase
    elif financial_status == 'Low Financial Status':
        return {"credit_limit_increase": 2000}  # $2,000 credit limit increase
    elif financial_status == 'New Low':
        return {"credit_limit_increase": 1000}  # $1,000 credit limit increase
    else:
        return {}

def loan_promotion_static(financial_status):
    if financial_status == 'High Financial Status':
        return {"interest_rate_loan": 3.0}  # 3% interest rate
    elif financial_status == 'Moderate Financial Status':
        return {"interest_rate_loan": 4.0}  # 4% interest rate
    elif financial_status == 'Low Financial Status':
        return {"interest_rate_loan": 5.0}  # 5% interest rate
    else:
        return {}

def insurance_promotion_static(financial_status):
    if financial_status == 'High Financial Status':
        return {"coverage": 5000}  # $5000 additional coverage
    elif financial_status == 'Moderate Financial Status':
        return {"coverage": 3000}  # $3000 basic coverage
    elif financial_status == 'Low Financial Status':
        return {"coverage": 1000}  # $1000 minimal coverage
    else:
        return {}

def investment_product_promotion_static(financial_status):
    if financial_status == 'High Financial Status':
        return {"return_rate": 7.0}  # 7% return rate
    elif financial_status == 'Moderate Financial Status':
        return {"return_rate": 5.0}  # 5% return rate
    elif financial_status == 'Low Financial Status':
        return {"return_rate": 3.0}  # 3% return rate
    else:
        return {}
class CampaignHistory:
    def __init__(self, campaign_database):
        self.campaign_database = campaign_database

    def get_conversion_rate(self, client, campaign_type):
        """
        Calculate the conversion rate as the number of successful responses
        divided by the total number of impressions for a given campaign type.
        """
        past_campaigns = [record for record in self.campaign_database if record['CLIENTNUM'] == client and record['Campaign_Type'] == campaign_type]

        if past_campaigns:
            total_impressions = len(past_campaigns)
            successful_responses = sum(1 for record in past_campaigns if record['Response_Status'] == 'successful')

            if total_impressions > 0:
                conversion_rate = successful_responses / total_impressions
                return conversion_rate
            else:
                return 0.0  # Avoid division by zero
        else:
            return None  # No past campaigns found
