import pandas as pd
import numpy as np

class DataManager:
    def __init__(self, df):
        self.df = df.copy()

    def add_data(self, new_data):
        # Set current date (Friday) for new entries
        current_time = pd.to_datetime('now').normalize()  # Set current time to the day, ignoring seconds
        new_data['Time'] = current_time  # Apply current date

        # Set CLIENTNUM as index for easy merging
        new_data.set_index('CLIENTNUM', inplace=True)

        # Remove churned customers from the current DataFrame
        churned_customers = new_data[new_data['Churned'] == 1].index
        self.df = self.df[~self.df['CLIENTNUM'].isin(churned_customers)]  # Remove churned customers

        # Set CLIENTNUM as index for the current dataframe
        self.df.set_index('CLIENTNUM', inplace=True)
        
        # Update existing records, ignoring churned customers
        self.df.update(new_data[new_data['Churned'] == 0])  
        
        # Append new records that don't exist in the current dataframe
        self.df = self.df.combine_first(new_data[new_data['Churned'] == 0])  

        # Reset index to return CLIENTNUM as a column
        self.df.reset_index(inplace=True)

        # Reorder columns to ensure 'Time' is right after 'CLIENTNUM'
        cols = list(self.df.columns)
        cols.insert(1, cols.pop(cols.index('Time')))  # Move 'Time' to right after 'CLIENTNUM'
        self.df = self.df[cols]

    def get(self):
        return self.df
    
class PercentileCalculator:
    def __init__(self, df):
        # Initialize with existing data
        self.df = df.copy()
        self.percentiles = {}

    def calculate_percentiles(self):
        # Calculate percentiles for the required columns
        self.percentiles['Credit_Score'] = self.df['Credit Score'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Outstanding_Loans'] = self.df['Outstanding Loans'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Balance'] = self.df['Balance'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Amt'] = self.df['Total_Trans_Amt'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Count'] = self.df['Total_Trans_Count'].quantile([0.2, 0.5, 0.8])

    def calculate_digital_capability(self, row):
        score = 0
        score += row['PhoneService']
        score += 1 if row['InternetService'] in [0, 1] else 0
        score += 1 if row['TechSupport'] == 2 else 0
        score += row['PaperlessBilling']
        score += 2 if row['PaymentMethod'] in [0, 1] else 1 if row['PaymentMethod'] == 2 else 0
        return True if score > 2 else False  # Return True for digitally capable, False for not capable

    def calculate_financial_status(self, row):
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

    def calculate_loyalty_score(self, row):
        # Loyalty is a composite score based on:
        # - Total_Trans_Amt: Scaled to a maximum of 3 points.
        # - Total_Trans_Count: Scaled to a maximum of 3 points.
        # - No_of_product: Heavy User (>4 products): 3 points, Moderate User (3-4 products): 2 points, Light User (<=2 products): 1 point

        trans_amt_score = 3 if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.8] else \
                          2 if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.5] else \
                          1 if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.2] else 0

        trans_count_score = 3 if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.8] else \
                            2 if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.5] else \
                            1 if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.2] else 0

        product_usage_score = 3 if row['No_of_product'] > 4 else 2 if 3 <= row['No_of_product'] <= 4 else 1

        return trans_amt_score + trans_count_score + product_usage_score

    def perform_feature_engineering(self):
        # Perform feature engineering to calculate Financial_Status, Loyalty, and Digital_Capability
        self.df['Financial_Status'] = self.df.apply(self.calculate_financial_status, axis=1)
        self.df['Loyalty'] = self.df.apply(self.calculate_loyalty_score, axis=1)
        self.df['Digital_Capability'] = self.df.apply(self.calculate_digital_capability, axis=1)

    def categorize_financial_status_and_loyalty(self):
        # Calculate percentiles for Financial_Status and Loyalty
        loyalty_percentiles = self.df['Loyalty'].quantile([0.2, 0.8])
        financial_status_percentiles = self.df['Financial_Status'].quantile([0.2, 0.8])

        # Categorize Financial_Status
        self.df['Financial_Status_Category'] = self.df['Financial_Status'].apply(
            lambda x: 'Low' if x <= financial_status_percentiles[0.2] else
            ('High' if x > financial_status_percentiles[0.8] else 'Moderate'))

        # Categorize Loyalty
        self.df['Loyalty_Category'] = self.df['Loyalty'].apply(
            lambda x: 'Low' if x <= loyalty_percentiles[0.2] else
            ('High' if x > loyalty_percentiles[0.8] else 'Moderate'))

    def get_featured_data(self):
        # Perform feature engineering and return a DataFrame with CLIENTNUM, Time, Financial_Status, Loyalty, and Digital_Capability
        self.perform_feature_engineering()
        self.categorize_financial_status_and_loyalty()
        return self.df[['CLIENTNUM','Time','Financial_Status_Category', 'Loyalty_Category', 'Digital_Capability']]
    
class Segmentation:
    def __init__(self, original_df, featured_df):
        """
        Initialize the Segmentation class with the original and featured data.
        
        :param original_df: The original DataFrame from the DataManager.
        :param featured_df: The DataFrame output from the PercentileCalculator.
        """
        self.original_df = original_df.copy()
        self.featured_df = featured_df.copy()

    def apply_segmentation_rule(self, row):
        """
        Apply segmentation rule based on the Financial_Status_Category and Loyalty_Category.
        
        :param row: A row from the featured_df DataFrame
        :return: Segment as per the predefined rules
        """
        financial_status = row['Financial_Status_Category']
        loyalty = row['Loyalty_Category']

        if financial_status == 'Low' and loyalty == 'Low':
            return 'Low Financial status, Low Loyalty'
        elif financial_status == 'High' and loyalty == 'High':
            return 'High Financial status, High Loyalty'
        elif financial_status == 'High' and loyalty in ['Moderate', 'Low']:
            return 'High Financial status, Low or Moderate Loyalty'
        elif financial_status in ['Moderate', 'Low'] and loyalty == 'High':
            return 'Low or Moderate Financial status, High Loyalty'
        else:
            return 'Moderate or Low Financial status, Moderate or Low Loyalty'

    def perform_segmentation(self):
        """
        Perform segmentation and append the SEGMENT column.
        """
        # Apply segmentation rule
        self.featured_df['SEGMENT'] = self.featured_df.apply(self.apply_segmentation_rule, axis=1)
        
        # Combine SEGMENT and DIGITAL_CAPABILITY from featured_df into the original_df
        self.original_df = self.original_df.merge(
            self.featured_df[['CLIENTNUM', 'SEGMENT','Digital_Capability']],
            on='CLIENTNUM',
            how='left'
        )

        

    def get_segment_result(self):
        """
        Get the result DataFrame with only CLIENTNUM,Time, SEGMENT, and DIGITAL_Capability (True/False).
        
        :return: A DataFrame containing CLIENTNUM, SEGMENT, and Digital_Capability.
        """
        return self.original_df[['CLIENTNUM','Time', 'SEGMENT', 'Digital_Capability']]

    def get_original_with_segment(self):
        """
        Get the original DataFrame with SEGMENT appended.
        
        :return: The original DataFrame with SEGMENT columns.
        """
        return self.original_df

class DynamicCustomerSegmentation:
    def __init__(self, initial_data):
        """
        Initialize the dynamic cluster segmentation with the initial dataset.
        
        :param initial_data: The original dataset, typically the output of DataManager.
        """
        self.data_manager = DataManager(initial_data)

    def process_new_data(self, new_data):
        """
        Process new data through the pipeline.
        
        :param new_data: The new batch of data to be processed.
        :return: A DataFrame with CLIENTNUM, SEGMENT, and DIGITAL_CAPABILITY by default.
        """
        # Step 1: Update the dataset using DataManager
        self.data_manager.add_data(new_data)
        updated_df = self.data_manager.get()

        # Step 2: Recalculate percentiles and perform feature engineering using PercentileCalculator
        percentile_calculator = PercentileCalculator(updated_df)
        percentile_calculator.calculate_percentiles()
        featured_data = percentile_calculator.get_featured_data()

        # Step 3: Perform segmentation using Segmentation class
        segmentation = Segmentation(original_df=updated_df, featured_df=featured_data)
        segmentation.perform_segmentation()

        # Step 4: Return the segmented result
        segmented_result = segmentation.get_segment_result()  # Simplified output with CLIENTNUM, SEGMENT, DIGITAL_CAPABILITY
        return segmented_result

    def get_full_original_with_segment(self):
        """
        Get the full original DataFrame with SEGMENT and DIGITAL_CAPABILITY appended.
        
        :return: The original DataFrame with the additional segmentation info.
        """
        segmentation = Segmentation(self.data_manager.get(), self.data_manager.get())  # Dummy instance for the getter
        return segmentation.get_original_with_segment()

