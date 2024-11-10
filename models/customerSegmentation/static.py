import pandas as pd
import numpy as np

"""
static.py

This module provides classes for static customer segmentation. It enables segmentation 
of customers based on predefined metrics such as loyalty, financial status, and 
digital capability, using a consistent dataset that is not continuously updated. 
The segmentation logic uses calculated percentiles to categorize customers 
into different segments based on their behavior and financial indicators.

Classes:
--------
- `CustomerSegmentation`: 
  - Manages the entire segmentation process, from calculating percentiles for key 
    metrics to assigning segments based on predefined rules.
"""

class CustomerSegmentation:
    """
    A class for performing customer segmentation based on financial status and loyalty metrics.
    
    Attributes:
    ----------
    df : pd.DataFrame
        The customer data on which segmentation is to be performed.
    percentiles : dict
        Dictionary storing percentile values for relevant features.

    Methods:
    -------
    calculate_initial_percentiles()
        Calculates percentiles for features required before creating Loyalty and Financial Status.
        
    calculate_final_percentiles()
        Calculates percentiles for Loyalty and Financial Status after they have been created.
        
    digital_capability(row)
        Determines digital capability of a customer based on service usage and billing method.
        
    financial_status(row)
        Calculates a financial score for each customer based on income, credit score, loans, and balance.
        
    transaction_behavior(row)
        Determines a customer's transaction behavior based on transaction amount and count.
        
    product_usage(row)
        Assigns a score based on product usage, indicating engagement level.
        
    loyalty_score(row)
        Calculates a loyalty score by combining transaction behavior and product usage.
        
    assign_loyalty_level(loyalty_score)
        Classifies loyalty into 'High', 'Moderate', or 'Low' based on percentile thresholds.
        
    assign_financial_status_level(financial_status_score)
        Classifies financial status into 'High', 'Moderate', or 'Low' based on percentile thresholds.
        
    assign_segment(row)
        Assigns a customer to a segment based on loyalty and financial status levels.
        
    perform_segmentation()
        Executes the full segmentation pipeline and returns key metrics and segment assignments.
        
    predict(new_data)
        Applies segmentation logic to new data.
    """
    def __init__(self, df):
        """
        Initializes the CustomerSegmentation class with customer data.

        Parameters:
        ----------
        df : pd.DataFrame
            The input data containing customer information.
        """
        self.df = df.copy()
        self.percentiles = {}

    def calculate_initial_percentiles(self):
        """
        Calculates percentiles for features required before Loyalty and Financial Status are created.
        
        This method initializes the percentiles for 'Credit Score', 'Outstanding Loans', 'Balance',
        'Total_Trans_Amt', and 'Total_Trans_Count' based on 20th, 50th, and 80th percentiles.
        """
        self.percentiles['Credit_Score'] = self.df['Credit Score'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Outstanding_Loans'] = self.df['Outstanding Loans'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Balance'] = self.df['Balance'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Amt'] = self.df['Total_Trans_Amt'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Count'] = self.df['Total_Trans_Count'].quantile([0.2, 0.5, 0.8])

    def calculate_final_percentiles(self):
        """
        Calculates percentiles for Loyalty and Financial Status after they have been created.
        
        This method initializes the percentiles for 'Loyalty' and 'Financial_Status' based on the 20th
        and 80th percentiles.
        """
        self.percentiles['Loyalty'] = self.df['Loyalty'].quantile([0.2, 0.8])
        self.percentiles['Financial_Status'] = self.df['Financial_Status'].quantile([0.2, 0.8])

    def digital_capability(self, row):
        """
        Determines digital capability of a customer based on service usage and billing method.

        Parameters:
        ----------
        row : pd.Series
            A row of customer data.

        Returns:
        -------
        bool
            True if the customer has sufficient digital capability, otherwise False.
        """
        score = 0
        score += row['PhoneService']
        score += 1 if row['InternetService'] in [0, 1] else 0
        score += 1 if row['TechSupport'] == 2 else 0
        score += row['PaperlessBilling']
        score += 2 if row['PaymentMethod'] in [0, 1] else 1 if row['PaymentMethod'] == 2 else 0
        return score > 2

    def financial_status(self, row):
        """
        Calculates a financial score for each customer based on income, credit score, loans, and balance.

        Parameters:
        ----------
        row : pd.Series
            A row of customer data.

        Returns:
        -------
        int
            A calculated financial status score.
        """
        score = 0
        # Income Category (strict rules)
        if row['Income_Category'] == '120 +':
            score += 3
        elif row['Income_Category'] == '80 - 120':
            score += 2
        elif row['Income_Category'] == '60 - 80':
            score += 1

        # Credit Score (percentile-based)
        if row['Credit Score'] > self.percentiles['Credit_Score'][0.8]:
            score += 3
        elif row['Credit Score'] > self.percentiles['Credit_Score'][0.5]:
            score += 2
        elif row['Credit Score'] > self.percentiles['Credit_Score'][0.2]:
            score += 1

        # Outstanding Loans (percentile-based)
        if row['Outstanding Loans'] < self.percentiles['Outstanding_Loans'][0.2]:
            score += 3
        elif row['Outstanding Loans'] < self.percentiles['Outstanding_Loans'][0.5]:
            score += 2
        elif row['Outstanding Loans'] < self.percentiles['Outstanding_Loans'][0.8]:
            score += 1

        # Balance (percentile-based)
        if row['Balance'] > self.percentiles['Balance'][0.8]:
            score += 3
        elif row['Balance'] > self.percentiles['Balance'][0.5]:
            score += 2
        elif row['Balance'] > self.percentiles['Balance'][0.2]:
            score += 1

        return score

    def transaction_behavior(self, row):
        """
        Determines a customer's transaction behavior based on transaction amount and count.

        Parameters:
        ----------
        row : pd.Series
            A row of customer data.

        Returns:
        -------
        int
            A transaction behavior score.
        """
        score = 0
        if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.8]:
            score += 3
        elif row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.5]:
            score += 2
        elif row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.2]:
            score += 1

        if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.8]:
            score += 3
        elif row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.5]:
            score += 2
        elif row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.2]:
            score += 1

        return score

    def product_usage(self, row):
        """
        Assigns a score based on product usage, indicating engagement level.

        Parameters:
        ----------
        row : pd.Series
            A row of customer data.

        Returns:
        -------
        int
            A product usage score.
        """
        if row['No_of_product'] > 4:
            return 3
        elif 3 <= row['No_of_product'] <= 4:
            return 2
        return 1

    def loyalty_score(self, row):
        """
        Calculates a loyalty score by combining transaction behavior and product usage.

        Parameters:
        ----------
        row : pd.Series
            A row of customer data.

        Returns:
        -------
        int
            A calculated loyalty score.
        """
        return self.transaction_behavior(row) + self.product_usage(row)

    def assign_loyalty_level(self, loyalty_score):
        """
        Classifies loyalty into 'High', 'Moderate', or 'Low' based on percentile thresholds.

        Parameters:
        ----------
        loyalty_score : int
            The loyalty score.

        Returns:
        -------
        str
            Loyalty level as 'High', 'Moderate', or 'Low'.
        """
        if loyalty_score > self.percentiles['Loyalty'][0.8]:
            return 'High'
        elif loyalty_score > self.percentiles['Loyalty'][0.2]:
            return 'Moderate'
        else:
            return 'Low'

    def assign_financial_status_level(self, financial_status_score):
        """
        Classifies financial status into 'High', 'Moderate', or 'Low' based on percentile thresholds.

        Parameters:
        ----------
        financial_status_score : int
            The financial status score.

        Returns:
        -------
        str
            Financial status level as 'High', 'Moderate', or 'Low'.
        """
        if financial_status_score > self.percentiles['Financial_Status'][0.8]:
            return 'High'
        elif financial_status_score > self.percentiles['Financial_Status'][0.2]:
            return 'Moderate'
        else:
            return 'Low'

    def assign_segment(self, row):
        """
        Assigns a customer to a segment based on loyalty and financial status levels.
        
        Parameters:
        ----------
        row : pd.Series
            A row of customer data containing 'Loyalty' and 'Financial_Status' scores.
        
        Returns:
        -------
        str
            The segment to which the customer is assigned, based on predefined classification rules.
        """
        # Assign loyalty and financial status levels
        loyalty_label = self.assign_loyalty_level(row['Loyalty'])
        financial_status_label = self.assign_financial_status_level(row['Financial_Status'])

        # Return segment based on the classification of loyalty and financial status
        if financial_status_label == 'Low' and loyalty_label == 'Low':
            return 'Low Financial status, Low Loyalty'
        elif financial_status_label == 'High' and loyalty_label == 'High':
            return 'High Financial status, High Loyalty'
        elif financial_status_label == 'High' and loyalty_label in ['Moderate', 'Low']:
            return 'High Financial status, Low or Moderate Loyalty'
        elif financial_status_label in ['Moderate', 'Low'] and loyalty_label == 'High':
            return 'Low or Moderate Financial status, High Loyalty'
        else:
            return 'Moderate or Low Financial status, Moderate or Low Loyalty'

    def perform_segmentation(self):
        """
        Executes the full segmentation pipeline on the current dataset.
        
        This method calculates initial percentiles, assigns loyalty and financial status scores,
        calculates their percentiles, and ultimately assigns each customer to a segment based on
        loyalty and financial status levels.
        
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing 'CLIENTNUM', 'Loyalty', 'Financial_Status', 'Segment', 
            and 'Digital_Capability' columns.
        """
        # Calculate percentiles for features before Loyalty and Financial Status
        self.calculate_initial_percentiles()

        self.df['Digital_Capability'] = self.df.apply(self.digital_capability, axis=1)
        self.df['Financial_Status'] = self.df.apply(self.financial_status, axis=1)
        self.df['Loyalty'] = self.df.apply(self.loyalty_score, axis=1)

        # After Loyalty and Financial_Status are created, calculate their percentiles
        self.calculate_final_percentiles()

        # Assign segment based on loyalty and financial status
        self.df['Segment'] = self.df.apply(self.assign_segment, axis=1)

        return self.df[['CLIENTNUM','Loyalty','Financial_Status','Segment', 'Digital_Capability']]

    def predict(self, new_data):
        """
        Applies segmentation logic to new data, enabling segmentation of previously unseen customers.
        
        This method calculates digital capability, financial status, and loyalty scores for each row in
        the new dataset, and then assigns a segment based on these values. Percentiles are recalculated
        for 'Loyalty' and 'Financial_Status' based on the new data.
        
        Parameters:
        ----------
        new_data : pd.DataFrame
            The new customer data to be segmented.
        
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing 'CLIENTNUM', 'Loyalty', 'Financial_Status', 'Segment', 
            and 'Digital_Capability' columns for the new data.
        """
        # Apply segmentation logic to new data
        new_data['Digital_Capability'] = new_data.apply(self.digital_capability, axis=1)
        new_data['Financial_Status'] = new_data.apply(self.financial_status, axis=1)
        new_data['Loyalty'] = new_data.apply(self.loyalty_score, axis=1)

        # Calculate percentiles for Loyalty and Financial Status based on new data
        self.calculate_final_percentiles()

        new_data['Segment'] = new_data.apply(self.assign_segment, axis=1)
        return new_data[['CLIENTNUM','Loyalty','Financial_Status','Segment', 'Digital_Capability']]