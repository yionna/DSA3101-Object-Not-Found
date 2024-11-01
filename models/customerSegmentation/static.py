import pandas as pd
import numpy as np

class CustomerSegmentation:
    def __init__(self, df):
        self.df = df.copy()
        self.percentiles = {}

    def calculate_initial_percentiles(self):
        # Calculate percentiles for features before Loyalty and Financial_Status are created
        self.percentiles['Credit_Score'] = self.df['Credit Score'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Outstanding_Loans'] = self.df['Outstanding Loans'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Balance'] = self.df['Balance'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Amt'] = self.df['Total_Trans_Amt'].quantile([0.2, 0.5, 0.8])
        self.percentiles['Total_Trans_Count'] = self.df['Total_Trans_Count'].quantile([0.2, 0.5, 0.8])

    def calculate_final_percentiles(self):
        # After Loyalty and Financial_Status have been created, calculate their percentiles
        self.percentiles['Loyalty'] = self.df['Loyalty'].quantile([0.2, 0.8])
        self.percentiles['Financial_Status'] = self.df['Financial_Status'].quantile([0.2, 0.8])

    def digital_capability(self, row):
        score = 0
        score += row['PhoneService']
        score += 1 if row['InternetService'] in [0, 1] else 0
        score += 1 if row['TechSupport'] == 2 else 0
        score += row['PaperlessBilling']
        score += 2 if row['PaymentMethod'] in [0, 1] else 1 if row['PaymentMethod'] == 2 else 0
        return score > 2

    def financial_status(self, row):
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
        if row['No_of_product'] > 4:
            return 3
        elif 3 <= row['No_of_product'] <= 4:
            return 2
        return 1

    def loyalty_score(self, row):
        return self.transaction_behavior(row) + self.product_usage(row)

    def assign_loyalty_level(self, loyalty_score):
        if loyalty_score > self.percentiles['Loyalty'][0.8]:
            return 'High'
        elif loyalty_score > self.percentiles['Loyalty'][0.2]:
            return 'Moderate'
        else:
            return 'Low'

    def assign_financial_status_level(self, financial_status_score):
        if financial_status_score > self.percentiles['Financial_Status'][0.8]:
            return 'High'
        elif financial_status_score > self.percentiles['Financial_Status'][0.2]:
            return 'Moderate'
        else:
            return 'Low'

    def assign_segment(self, row):
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
        # Calculate percentiles for features before Loyalty and Financial Status
        self.calculate_initial_percentiles()

        self.df['Digital_Capability'] = self.df.apply(self.digital_capability, axis=1)
        self.df['Financial_Status'] = self.df.apply(self.financial_status, axis=1)
        self.df['Loyalty'] = self.df.apply(self.loyalty_score, axis=1)

        # After Loyalty and Financial_Status are created, calculate their percentiles
        self.calculate_final_percentiles()

        # Assign segment based on loyalty and financial status
        self.df['Segment'] = self.df.apply(self.assign_segment, axis=1)

        return self.df[['Segment', 'Digital_Capability']]

    def predict(self, new_data):
        # Apply segmentation logic to new data
        new_data['Digital_Capability'] = new_data.apply(self.digital_capability, axis=1)
        new_data['Financial_Status'] = new_data.apply(self.financial_status, axis=1)
        new_data['Loyalty'] = new_data.apply(self.loyalty_score, axis=1)

        # Calculate percentiles for Loyalty and Financial Status based on new data
        self.calculate_final_percentiles()

        new_data['Segment'] = new_data.apply(self.assign_segment, axis=1)
        return new_data[['Segment', 'Digital_Capability']]