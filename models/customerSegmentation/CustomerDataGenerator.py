import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

class CustomerDataGenerator:
    """
    A class to generate and modify synthetic customer data with configurable churn rate,
    campaign effectiveness, and customer satisfaction levels. Supports updating existing 
    data and generating new customer entries with timestamps.
    
    Attributes:
    -----------
    churn_rate : float
        Probability of customer churn between 0 and 1.
    campaign_effectiveness : float
        Factor influencing customer engagement metrics. 
    customer_satisfaction : float
        Factor impacting customer satisfaction features.
    current : pandas.DataFrame
        Base dataset representing current customers.
    new : pandas.DataFrame
        DataFrame containing modified and newly generated data.
    time : datetime
        Timestamp starting from '2024-01-08' and incremented weekly.
    """

    def __init__(self, initial_data, churn_rate, campaign_effectiveness, customer_satisfaction):
        """
        Initialize the CustomerDataGenerator with initial data and configuration parameters.
        
        Parameters:
        -----------
        initial_data : pd.DataFrame
            DataFrame representing the initial dataset of customers.
        churn_rate : float
            Probability of customer churn, between 0 and 1.
        campaign_effectiveness : float
            Factor influencing customer engagement.
        customer_satisfaction : float
            Factor impacting customer satisfaction features.
        """
        self.churn_rate = churn_rate
        self.campaign_effectiveness = campaign_effectiveness
        self.customer_satisfaction = customer_satisfaction
        self.fake = Faker()
        self.current = initial_data.copy()
        self.new = pd.DataFrame()
        self.time = datetime(2024, 1, 8)

    def _generate_unique_clientnum(self):
        """
        Generate a unique CLIENTNUM that doesn't exist in the current dataset.
        
        Returns:
        --------
        int
            A unique CLIENTNUM.
        """
        while True:
            clientnum = np.random.randint(1, 999999999)
            if clientnum not in self.current['CLIENTNUM'].values:
                return clientnum

    def _adjust_value(self, value, max_value, min_value=1):
        """
        Adjust a value based on campaign effectiveness and customer satisfaction.
        
        Parameters:
        -----------
        value : int
            The original value to adjust.
        max_value : int
            Maximum possible adjusted value.
        min_value : int, optional
            Minimum possible adjusted value (default is 1).
        
        Returns:
        --------
        int
            Adjusted value.
        """
        if self.campaign_effectiveness < 0 or self.customer_satisfaction < 0:
            factor = 1 - abs(self.campaign_effectiveness * 0.2 + self.customer_satisfaction * 0.2)
        else:
            factor = 1 + self.campaign_effectiveness * 0.5 * self.customer_satisfaction * 0.5
        return max(min(int(value * factor), max_value), min_value)

    def update_existing_data(self):
        """
        Update a subset of customer data based on churn rate, effectiveness, and satisfaction.
        Modified data is stored in `self.new`.
        """
        num_to_update = int(np.random.uniform(0.0, 0.5) * len(self.current))
        num_churned = int(self.churn_rate * num_to_update)
        update_clients = np.random.choice(self.current.index, size=num_to_update, replace=False)
        churned_clients = np.random.choice(update_clients, size=num_churned, replace=False)
        
        modified_data = []
        for idx in update_clients:
            row = self.current.loc[idx].copy()
            row['No_of_product'] = self._adjust_value(row['No_of_product'], 6)
            row['Total_Trans_Amt'] = self._adjust_value(row['Total_Trans_Amt'], 10000)
            row['Total_Trans_Count'] = self._adjust_value(row['Total_Trans_Count'], 150)
            row['Churned'] = 1 if idx in churned_clients else 0
            row['Time'] = self.time
            modified_data.append(row)
        self.new = pd.concat([self.new, pd.DataFrame(modified_data)], ignore_index=True)

    def generate_new_customers(self):
        """
        Generate new customer entries based on campaign effectiveness.
        New data is added to `self.new`.
        """
        num_new = int(10000 * (self.campaign_effectiveness * 0.5)) if self.campaign_effectiveness > 0 else 0
        if num_new > 0:
            new_customers = [{
                'CLIENTNUM': self._generate_unique_clientnum(),
                'Income_Category': self.fake.random_int(min=0, max=4),
                'No_of_product': self._adjust_value(self.fake.random_int(min=1, max=3), 6),
                'Total_Trans_Amt': self._adjust_value(self.fake.random_int(min=500, max=2000), 10000),
                'Total_Trans_Count': self._adjust_value(self.fake.random_int(min=10, max=50), 150),
                'Credit Score': self.fake.random_int(min=300, max=850),
                'Outstanding Loans': self.fake.random_int(min=0, max=50000),
                'Balance': self.fake.random_int(min=0, max=300000),
                'PhoneService': self.fake.random_int(min=0, max=1),
                'InternetService': self.fake.random_int(min=0, max=2),
                'TechSupport': self.fake.random_int(min=0, max=2),
                'PaperlessBilling': self.fake.random_int(min=0, max=1),
                'PaymentMethod': self.fake.random_int(min=0, max=3),
                'Churned': 0,
                'Time': self.time
            } for _ in range(num_new)]
            new_data_df = pd.DataFrame(new_customers)
            self.new = pd.concat([self.new, new_data_df], ignore_index=True)

    def reset(self, new_df, churn_rate=0, campaign_effectiveness=0, customer_satisfaction=0):
        """
        Reset the generator with a new DataFrame and update configuration.
        
        Parameters:
        -----------
        new_df : pd.DataFrame
            New dataset to use as the base dataset.
        churn_rate : float, optional
            Incremental churn rate (default is 0).
        campaign_effectiveness : float, optional
            Incremental campaign effectiveness (default is 0).
        customer_satisfaction : float, optional
            Incremental customer satisfaction (default is 0).
        """
        self.current = new_df.copy()
        self.new = pd.DataFrame()
        self.time += timedelta(days=7)
        self.churn_rate = min(max(self.churn_rate + churn_rate, 0), 1)
        self.campaign_effectiveness = min(max(self.campaign_effectiveness + campaign_effectiveness, -1), 1)
        self.customer_satisfaction = min(max(self.customer_satisfaction + customer_satisfaction, -1), 1)

    def get_new_data(self):
        """
        Retrieve the DataFrame with modified and newly generated customer data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing modified and new customer data.
        """
        self.new.reset_index(drop=True, inplace=True)
        return self.new