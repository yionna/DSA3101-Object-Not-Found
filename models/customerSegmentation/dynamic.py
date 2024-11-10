import pandas as pd
import numpy as np
"""
dynamic.py

This module contains classes for a dynamic customer segmentation system. The system 
is designed to manage and analyze customer data, enabling real-time segmentation 
based on updated information. The module consists of three main classes:

- `DataManager`: Manages customer records, allowing for the addition, update, and 
  removal of customer data based on churn status.
  
- `PercentileCalculator`: Calculates percentiles for key metrics and derives additional 
  features such as financial status, loyalty, and digital capability. This class 
  supports dynamic data processing by calculating percentile-based scores.

- `Segmentation`: Segments customers based on loyalty and financial capability, using 
  predefined rules to classify customers into groups for targeted marketing 
  and engagement strategies.

- `DynamicCustomerSegmentation`: Integrates the `DataManager`, `PercentileCalculator`, 
  and `Segmentation` classes, providing a streamlined workflow for dynamically 
  processing new data, recalculating metrics, and assigning segments to customers 
  on a regular basis. It is designed to support real-time decision-making 
  by adapting to continuously updated customer data.

The module enables businesses to identify high-value customers, understand loyalty 
patterns, and implement personalized engagement strategies based on real-time data.

Example Usage:
--------------
```python
# Initialize with an initial dataset
dynamic_segmentation = DynamicCustomerSegmentation(initial_data)

# Process a new batch of customer data and retrieve segmented output
segmented_result = dynamic_segmentation.process_new_data(new_data)

# Access the full updated dataset with segments appended
full_data_with_segments = dynamic_segmentation.get_full_original_with_segment()
"""
class DataManager:
    """
    DataManager is responsible for managing customer data updates, including
    adding new records, updating existing records, and handling customer churn.
    This class serves as the primary interface for modifying and retrieving
    the current state of customer data, keeping it up-to-date for subsequent
    analysis and segmentation.

    Attributes:
    -----------
    df : pandas.DataFrame
        A DataFrame that holds the initial customer data, including key attributes 
        required for segmentation and analysis.

    Methods:
    --------
    add_data(new_data):
        Updates the existing dataset by adding new entries, updating existing records,
        and removing churned customers as indicated by the new data. Each entry is
        timestamped to reflect the latest update.

    get():
        Retrieves the current state of the managed customer dataset after updates,
        with churned customers removed and new data incorporated.
    """
    def __init__(self, df):
        self.df = df.copy()

    def add_data(self, new_data):
        """
        Update the current dataset by incorporating new customer data, updating
        existing records, and removing records for churned customers.

        Parameters:
        -----------
        new_data : pandas.DataFrame
            A DataFrame containing the new data to be added or updated. Includes
            a 'CLIENTNUM' identifier, a 'Churned' column to indicate churn status,
            and a 'Time' column to timestamp the updates.

        Operations:
        -----------
        - Sets a timestamp (Friday) on new entries.
        - Removes records where 'Churned' is marked as 1.
        - Updates records where 'CLIENTNUM' exists in both current and new datasets.
        - Appends records for new customers.
        - Ensures 'Time' is placed directly after 'CLIENTNUM' for clarity.

        Returns:
        --------
        None
        """
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
        """
        Retrieve the current customer dataset after updates have been applied.
        
        Returns:
        --------
        pandas.DataFrame
            The updated customer DataFrame with churned customers removed,
            new records appended, and 'Time' column organized for readability.
        """
        return self.df
    
class PercentileCalculator:
    """
    The PercentileCalculator class is designed to calculate customer features
    related to financial capability, loyalty, and digital engagement based on
    percentile thresholds. This class provides methods for creating feature scores,
    segmenting customers based on loyalty and financial status, and generating a
    DataFrame with the categorized attributes for further analysis or segmentation.

    Attributes:
    -----------
    df : pandas.DataFrame
        The DataFrame containing customer data, including fields necessary
        for financial and loyalty scoring calculations.
    percentiles : dict
        A dictionary to store calculated percentiles for key financial and
        transactional attributes.

    Methods:
    --------
    calculate_percentiles():
        Computes the 20th, 50th, and 80th percentiles for relevant financial
        and transactional attributes such as credit score and balance.

    calculate_digital_capability(row):
        Calculates a digital capability score for each customer based on
        selected service usage attributes.

    calculate_financial_status(row):
        Calculates a financial status score for each customer based on income
        category, credit score, outstanding loans, and balance percentiles.

    calculate_loyalty_score(row):
        Calculates a loyalty score based on transaction amounts, transaction counts,
        and product usage, using percentile-based thresholds.

    perform_feature_engineering():
        Generates Financial_Status, Loyalty, and Digital_Capability scores for
        each customer and appends these columns to the main DataFrame.

    categorize_financial_status_and_loyalty():
        Categorizes customers into low, moderate, or high categories for both
        financial status and loyalty based on the calculated scores.

    get_featured_data():
        Performs feature engineering and categorization, returning a DataFrame
        containing CLIENTNUM, Time, Financial_Status, Loyalty, and Digital_Capability.
    """
    def __init__(self, df):
        # Initialize with existing data
        self.df = df.copy()
        self.percentiles = {}

    def calculate_percentiles(self):
        """
        Calculates the 20th, 50th, and 80th percentiles for key attributes:
        Credit Score, Outstanding Loans, Balance, Total Transaction Amount,
        and Total Transaction Count. These percentiles are used to classify
        financial status and loyalty levels in later methods.
        """
        self.percentiles['Credit_Score'] = self.df['Credit Score'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Outstanding_Loans'] = self.df['Outstanding Loans'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Balance'] = self.df['Balance'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Amt'] = self.df['Total_Trans_Amt'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Count'] = self.df['Total_Trans_Count'].quantile([0.2, 0.5, 0.8])

    def calculate_digital_capability(self, row):
        """
        Calculates a digital capability score based on the customer's use of
        digital services. Returns True if the customer is considered digitally
        capable, based on a cumulative score threshold.

        Parameters:
        -----------
        row : pandas.Series
            A single row of customer data.

        Returns:
        --------
        bool
            True if the customer is digitally capable; False otherwise.
        """
        score = 0
        score += row['PhoneService']
        score += 1 if row['InternetService'] in [0, 1] else 0
        score += 1 if row['TechSupport'] == 2 else 0
        score += row['PaperlessBilling']
        score += 2 if row['PaymentMethod'] in [0, 1] else 1 if row['PaymentMethod'] == 2 else 0
        return True if score > 2 else False  # Return True for digitally capable, False for not capable

    def calculate_financial_status(self, row):
        """
        Computes a financial status score based on income category and percentile-based
        thresholds for credit score, outstanding loans, and balance. Higher scores
        indicate higher financial capability.

        Parameters:
        -----------
        row : pandas.Series
            A single row of customer data.

        Returns:
        --------
        int
            A score representing the customer's financial status.
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

    def calculate_loyalty_score(self, row):
        """
        Computes a loyalty score based on percentile rankings for transaction
        amount, transaction count, and product usage. Higher scores indicate
        greater customer loyalty.
        
        Loyalty is a composite score based on:
        - Total_Trans_Amt: Scaled to a maximum of 3 points.
        - Total_Trans_Count: Scaled to a maximum of 3 points.
        - No_of_product: Heavy User (>4 products): 3 points, Moderate User (3-4 products): 2 points, Light User (<=2 products): 1 point

        Parameters:
        -----------
        row : pandas.Series
            A single row of customer data.

        Returns:
        --------
        int
            A score representing the customer's loyalty.
        """
        trans_amt_score = 3 if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.8] else \
                          2 if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.5] else \
                          1 if row['Total_Trans_Amt'] > self.percentiles['Total_Trans_Amt'][0.2] else 0

        trans_count_score = 3 if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.8] else \
                            2 if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.5] else \
                            1 if row['Total_Trans_Count'] > self.percentiles['Total_Trans_Count'][0.2] else 0

        product_usage_score = 3 if row['No_of_product'] > 4 else 2 if 3 <= row['No_of_product'] <= 4 else 1

        return trans_amt_score + trans_count_score + product_usage_score

    def perform_feature_engineering(self):
        """
        Calculates feature scores for Financial_Status, Loyalty, and Digital_Capability.

        - Financial_Status: A composite score based on income, credit score, outstanding loans,
        and balance, indicating the financial health of the customer.
        - Loyalty: A score representing customer engagement level, derived from transaction
         behavior and product usage.
        - Digital_Capability: Boolean score to represent the digital engagement level
         of the customer.

        These scores are added as new columns in the DataFrame.
        """
        # Perform feature engineering to calculate Financial_Status, Loyalty, and Digital_Capability
        self.df['Financial_Status'] = self.df.apply(self.calculate_financial_status, axis=1)
        self.df['Loyalty'] = self.df.apply(self.calculate_loyalty_score, axis=1)
        self.df['Digital_Capability'] = self.df.apply(self.calculate_digital_capability, axis=1)

    def categorize_financial_status_and_loyalty(self):
        """
        Categorizes Financial_Status and Loyalty scores into 'Low', 'Moderate', or 'High'
        categories based on calculated percentiles (20th and 80th).

        - Financial_Status_Category: Categorizes financial status into 'Low', 'Moderate', or 'High' based on 20th and 80th percentile thresholds.
        - Loyalty_Category: Categorizes loyalty into 'Low', 'Moderate', or 'High' based on 20th and 80th percentile thresholds.
        """
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
        """
        Returns a DataFrame with selected engineered features and categories for further analysis.

        - Calls perform_feature_engineering() to compute feature scores.
        - Calls categorize_financial_status_and_loyalty() to assign categorical values based on percentile thresholds.

        Returns:
        --------
        pandas.DataFrame:
            A DataFrame containing CLIENTNUM, Time, Financial_Status_Category, Loyalty_Category,
            and Digital_Capability.
        """
        # Perform feature engineering and return a DataFrame with CLIENTNUM, Time, Financial_Status, Loyalty, and Digital_Capability
        self.perform_feature_engineering()
        self.categorize_financial_status_and_loyalty()
        return self.df[['CLIENTNUM','Time','Financial_Status_Category', 'Loyalty_Category', 'Digital_Capability']]
    
class Segmentation:
    """
    The Segmentation class is responsible for categorizing customers into segments based on 
    their financial status and loyalty categories. This segmentation enables targeted marketing 
    and personalized engagement strategies.

    Parameters:
    ----------
    original_df : pandas.DataFrame
        The original customer DataFrame output from the DataManager, containing all customer data.
    featured_df : pandas.DataFrame
        The DataFrame output from PercentileCalculator, with additional columns such as 
        Financial_Status_Category, Loyalty_Category, and Digital_Capability.

    Methods:
    -------
    apply_segmentation_rule(row):
        Assigns a segment to each customer based on their Financial_Status_Category and 
        Loyalty_Category using predefined rules.

    perform_segmentation():
        Executes the segmentation process, creating a new 'SEGMENT' column in the 
        featured_df DataFrame, and merges the result back into original_df.

    get_segment_result():
        Returns a simplified DataFrame containing CLIENTNUM, Time, SEGMENT, and Digital_Capability.

    get_original_with_segment():
        Returns the original DataFrame with SEGMENT and Digital_Capability columns appended, 
        providing a complete view of customer data with segmentation applied.
    """
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
    """
    The DynamicCustomerSegmentation class orchestrates the end-to-end process of managing, 
    updating, and segmenting customer data. It integrates data management, percentile 
    calculation, and segmentation functionality, allowing for real-time updates with each 
    new data batch. This dynamic approach enables ongoing segmentation based on updated 
    financial and loyalty metrics.

    Parameters:
    ----------
    initial_data : pandas.DataFrame
        The initial dataset, typically representing the existing customer data,
        provided to initialize the segmentation model.

    Methods:
    -------
    process_new_data(new_data):
        Updates the dataset with new data, recalculates percentiles, and performs 
        feature engineering and segmentation. Returns a simplified result containing 
        CLIENTNUM, SEGMENT, and DIGITAL_CAPABILITY.

    get_full_original_with_segment():
        Retrieves the complete original dataset with SEGMENT and DIGITAL_CAPABILITY 
        columns appended, providing a full view of the updated and segmented data.
    """
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

