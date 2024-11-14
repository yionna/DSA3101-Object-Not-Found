import pandas as pd

class EarlyWarningSystem:
    def __init__(self, churn_status_threshold=1, financial_status_threshold=4.0, loyalty_threshold=3):
        self.churn_status_threshold = churn_status_threshold
        self.financial_status_threshold = financial_status_threshold
        self.loyalty_threshold = loyalty_threshold

    def apply_warning(self, demographic_db):
        """
        Apply the early warning system logic to the demographic database.
        """
        demographic_db['Alert'] = demographic_db.apply(
            lambda row: 'High risk of churn - follow up needed' if (
                row['Churn'] >= self.churn_status_threshold or
                row['Financial_Status'] < self.financial_status_threshold or
                row['Loyalty'] < self.loyalty_threshold
            ) else 'Low risk',
            axis=1
        )
        return demographic_db

    def get_customers_to_update(self, previous_db, updated_db):
        """
        Identify customers who have changed from high risk to low risk and vice versa.

        Parameters:
        previous_db (DataFrame): The previous demographic database with alert statuses.
        updated_db (DataFrame): The updated demographic database with new alert statuses.

        Returns:
        Tuple[DataFrame, DataFrame]: Two DataFrames, one for customers who changed from 
                                     high risk to low risk and one for customers who changed 
                                     from low risk to high risk.
        """
        # Merge previous and updated databases on 'CustomerID'
        merged_db = previous_db[['CLIENTNUM', 'Alert']].merge(
            updated_db[['CLIENTNUM', 'Alert']],
            on='CLIENTNUM',
            suffixes=('_prev', '_new')
        )

        # Identify customers who went from high risk to low risk
        high_to_low_risk = merged_db[
            (merged_db['Alert_prev'] == 'High risk of churn - follow up needed') &
            (merged_db['Alert_new'] == 'Low risk')
        ]

        # Identify customers who went from low risk to high risk
        low_to_high_risk = merged_db[
            (merged_db['Alert_prev'] == 'Low risk') &
            (merged_db['Alert_new'] == 'High risk of churn - follow up needed')
        ]

        return high_to_low_risk, low_to_high_risk


