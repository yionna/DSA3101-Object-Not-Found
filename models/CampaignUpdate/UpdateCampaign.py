from datetime import datetime, timedelta
import pandas as pd

class CampaignSystem:
    def __init__(self, campaign_database):
        # Load the campaign data
        self.campaign_database = pd.read_csv(campaign_database)
        # Ensure that EndDate is parsed as a datetime object
        self.campaign_database['EndDate'] = pd.to_datetime(self.campaign_database['EndDate'])

    def generate_campaign_id(self, product, campaign_type, segment, date=None):
        """
        Generates a unique campaign ID based on product, campaign type, and current date.
        """
        if date is None:
            date = datetime.now()
        return f"{product}_{campaign_type}_{date.strftime('%Y%m%d')}_{segment}" #separate campaign by segment

    def get_campaign_duration(self, campaign_type):
        """
        Returns the campaign duration in days based on the campaign type.
        """
        if campaign_type == 'retention':
            return 30  # 1 month
        elif campaign_type == 'consideration':
            return 60  # 2 months
        elif campaign_type == 'conversion':
            return 90  # 3 months
        else:
            return 30  # Default to 1 month

    def check_active_campaign(self, clientnum, campaign_type):
        """
        Checks if a customer is currently involved in an active campaign of the same type
        by verifying if the EndDate has not passed.
        """
        active_campaigns = self.campaign_database[
            (self.campaign_database['CLIENTNUM'] == clientnum) &
            (self.campaign_database['CampaignType'] == campaign_type) &
            (self.campaign_database['EndDate'] >= datetime.now())
        ]
        return not active_campaigns.empty

    def update_campaign_database(self, customer_product_list):
        """
        Updates the campaign database based on the given list of customer and product pairs.
        Returns the updated campaign database as a DataFrame.
        """
        updated_campaigns = self.campaign_database.copy()

        for clientnum, product in customer_product_list:
            # Determine campaign type based on product
            if product in ['investments','personal insurance', 'commercial insurance',
                           'personal loans', 'commercial loans']:  # Conversion offers
                campaign_type = 'conversion'
            elif product in ['creditcard', 'saving accounts']:  # Consideration offers
                campaign_type = 'consideration'
            else:  # Retention offers for other products
                campaign_type = 'retention'

            # Check if the customer is already in an active campaign of the same type
            if not self.check_active_campaign(clientnum, campaign_type):
                # Generate a new campaign ID
                campaign_id = self.generate_campaign_id(product, campaign_type, updated_campaigns['segment'])

                # Determine start and end dates for the campaign
                start_date = datetime.now()
                end_date = start_date + timedelta(days=self.get_campaign_duration(campaign_type))

                # Create a new campaign entry
                new_campaign = {
                    'CLIENTNUM': clientnum,
                    'CampaignID': campaign_id,
                    'Product': product,
                    'CampaignType': campaign_type,
                    'StartDate': start_date.strftime('%Y-%m-%d'),
                    'EndDate': end_date.strftime('%Y-%m-%d'),
                    'Status': 'Active',
                    'ResponseStatus': 'Unknown',
                    'NumberOfImpressions': 0,
                    'NumberOfClicks': 0
                }

                # Append the new campaign to the updated campaign DataFrame
                updated_campaigns = pd.concat(
                    [updated_campaigns, pd.DataFrame([new_campaign])],
                    ignore_index=True
                )

        return updated_campaigns
