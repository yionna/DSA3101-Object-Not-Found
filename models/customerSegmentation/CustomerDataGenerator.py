from faker import Faker
import numpy as np
import pandas as pd
from datetime import datetime

class CustomerDataGenerator:
    """
    A class to generate synthetic customer data with configurable churn rate, campaign effectiveness, and customer satisfaction levels.

    This class simulates customer behavior by generating synthetic data for existing and new customers. The data generated can reflect varying 
    levels of campaign effectiveness and customer satisfaction, allowing for controlled testing of customer engagement scenarios. The generated
    data includes features such as income category, number of products, transaction amounts, transaction counts, churn status, and timestamps.

    Attributes:
    ----------
    churn_rate : float
        Probability of churn among existing customers, ranging from 0 to 1, where 1 represents all existing customers churned.
    campaign_effectiveness : float
        Scale factor for campaign effectiveness, influencing customer engagement metrics (e.g., product usage and transaction frequency).
        Ranges from 0 to 1, with higher values indicating greater effectiveness.
    customer_satisfaction : float
        Scale factor for customer satisfaction, impacting features such as transaction amount and count. Ranges from 0 to 1, with higher 
        values indicating higher satisfaction.

    Methods:
    -------
    _adjust_based_on_campaign(value, max_increase):
        Private helper method to adjust values based on campaign effectiveness and customer satisfaction.
    generate_data(existing_df, num_records=1000, timestamp=None):
        Generates synthetic data with a mix of existing and new customer records. Allows for a consistent timestamp across records, 
        simulating a specific point in time.
    
    Example:
    -------
    # Import necessary libraries
    import pandas as pd
    from datetime import datetime
    
    # Load the existing dataset
    banking_df = pd.read_csv("../data/processed/banking_behaviour_preference.csv")

    # Initialize the CustomerDataGenerator for high effectiveness and satisfaction
    high_generator = CustomerDataGenerator(churn_rate=0.1, campaign_effectiveness=0.9, customer_satisfaction=0.9)
    
    # Set the timestamp for high effectiveness and satisfaction
    high_timestamp = datetime(2024, 1, 1)
    
    # Generate the high-effectiveness dataset
    high_data = high_generator.generate_data(banking_df, num_records=10000, timestamp=high_timestamp)
    """
    def __init__(self, churn_rate=0.1, campaign_effectiveness=0.5, customer_satisfaction=0.5):
        """
        Initialize the generator with churn rate, campaign effectiveness, and customer satisfaction.
        
        :param churn_rate: Probability of churn for existing clients
        :param campaign_effectiveness: Scale factor (0-1) for campaign's effectiveness on features
        :param customer_satisfaction: Scale factor (0-1) for customer satisfaction effect on features
        """
        self.churn_rate = churn_rate
        self.campaign_effectiveness = campaign_effectiveness
        self.customer_satisfaction = customer_satisfaction
        self.fake = Faker()
    
    def _adjust_based_on_campaign(self, value, max_increase):
        """Adjust values based on campaign effectiveness and customer satisfaction."""
        increase_factor = 1 + self.campaign_effectiveness * self.customer_satisfaction  # Value between 1 and 2
        return min(int(value * increase_factor), max_increase)

    def generate_data(self, existing_df, num_records, timestamp=None):
        """
        Generate synthetic customer data with a fixed timestamp.
        
        :param existing_df: DataFrame containing existing customer data
        :param num_records: Total number of records to generate
        :param timestamp: Fixed datetime to apply to all records in the generated data
        :return: DataFrame with generated customer data
        """
        # Set the timestamp to the current datetime if none is provided
        if timestamp is None:
            timestamp = datetime.now()

        # Get existing client numbers
        existing_clients = existing_df['CLIENTNUM'].values

        # Determine number of existing and new records
        num_existing = int(np.random.uniform(0.0, 0.8) * num_records)
        num_existing = min(num_existing, len(existing_clients))  # Ensure we do not exceed the actual number
        num_new = num_records - num_existing

        # Select random existing clients for updates
        updated_client_nums = np.random.choice(existing_clients, size=num_existing, replace=False)
        new_client_nums = np.arange(existing_clients.max() + 1, existing_clients.max() + 1 + num_new)

        # Create updated records for existing clients
        updated_data = []
        for client in updated_client_nums:
            updated_data.append({
                'CLIENTNUM': client,
                'Income_Category': self.fake.random_int(min=0, max=4),  # Income category as integer
                'No_of_product': self._adjust_based_on_campaign(self.fake.random_int(min=1, max=3), 6),
                'Total_Trans_Amt': self._adjust_based_on_campaign(self.fake.random_int(min=500, max=2000), 10000),
                'Total_Trans_Count': self._adjust_based_on_campaign(self.fake.random_int(min=10, max=50), 150),
                'Credit Score': self.fake.random_int(min=300, max=850),
                'Outstanding Loans': self.fake.random_int(min=0, max=50000),
                'Balance': self.fake.random_int(min=0, max=300000),
                'PhoneService': self.fake.random_int(min=0, max=1),
                'InternetService': self.fake.random_int(min=0, max=2),
                'TechSupport': self.fake.random_int(min=0, max=2),
                'PaperlessBilling': self.fake.random_int(min=0, max=1),
                'PaymentMethod': self.fake.random_int(min=0, max=3),
                'Churned': 0,  # Initially set as not churned
                'Time': timestamp  # Fixed timestamp for each record
            })
        
        # Apply churn rate only to the updated (existing) customers
        num_churned = int(self.churn_rate * num_existing)
        churned_clients = np.random.choice(range(num_existing), size=num_churned, replace=False)
        
        for i in churned_clients:
            updated_data[i]['Churned'] = 1  # Mark these clients as churned

        # Create new customer records (with no churn)
        new_data = []
        for client in new_client_nums:
            new_data.append({
                'CLIENTNUM': client,
                'Income_Category': self.fake.random_int(min=0, max=4),  # Income category as integer
                'No_of_product': self._adjust_based_on_campaign(self.fake.random_int(min=1, max=3), 6),
                'Total_Trans_Amt': self._adjust_based_on_campaign(self.fake.random_int(min=500, max=2000), 10000),
                'Total_Trans_Count': self._adjust_based_on_campaign(self.fake.random_int(min=10, max=50), 150),
                'Credit Score': self.fake.random_int(min=300, max=850),
                'Outstanding Loans': self.fake.random_int(min=0, max=50000),
                'Balance': self.fake.random_int(min=0, max=300000),
                'PhoneService': self.fake.random_int(min=0, max=1),
                'InternetService': self.fake.random_int(min=0, max=2),
                'TechSupport': self.fake.random_int(min=0, max=2),
                'PaperlessBilling': self.fake.random_int(min=0, max=1),
                'PaymentMethod': self.fake.random_int(min=0, max=3),
                'Churned': 0,  # New customers are not churned
                'Time': timestamp  # Fixed timestamp for each record
            })
        
        # Combine updated and new data into one DataFrame
        all_data = pd.DataFrame(updated_data + new_data)

        return all_data