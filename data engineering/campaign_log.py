def create_campaign_log(campaign_database):
    """
    Creates a campaign log by grouping the campaign database by CampaignID and calculating
    the total number of impressions, clicks, and conversion rate for each campaign.
    
    Parameters:
    campaign_database (DataFrame): The original campaign database.
    
    Returns:
    DataFrame: A new campaign log DataFrame with aggregated metrics and default settings.
    """
    # Group by CampaignID and aggregate the number of impressions, clicks, and responses
    campaign_log = campaign_database.groupby('CampaignID').agg(
        TotalImpressions=('NumberOfImpressions', 'sum'),
        TotalClicks=('NumberOfClicks', 'sum'),
        TotalResponses=('ResponseStatus', lambda x: (x == 'Subscribed').sum())
    ).reset_index()

    # Calculate conversion rate and click-through rate (CTR)
    campaign_log['ConversionRate'] = campaign_log['TotalResponses'] / campaign_log['TotalImpressions']
    campaign_log['ClickThroughRate'] = campaign_log['TotalClicks'] / campaign_log['TotalImpressions']

    # Fill NaN values with 0 for rates where TotalImpressions might be 0
    campaign_log['ConversionRate'].fillna(0, inplace=True)
    campaign_log['ClickThroughRate'].fillna(0, inplace=True)

    # Add default channel, frequency, and timing
    nrows = len(campaign_log)
    campaign_log['ChosenChannel'] = ['Default'] * nrows
    campaign_log['ChosenFrequency'] = [1] * nrows
    campaign_log['ChosenTiming'] = [6] * nrows
    
    return campaign_log
