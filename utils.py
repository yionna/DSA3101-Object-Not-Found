# for generation of campaign data
def generate_synthetic_features_campaign(df, distribution, feature):
    synthetic_feature = []
    
    for _, row in df.iterrows():
        features = tuple(row[common_features])
        try:
            sample = np.random.choice(
                distribution.columns,
                p=distribution.loc[features].values
            )
        except KeyError: #incase the feature combination is missing
            sample = np.random.choice(distribution.columns)
            
        synthetic_feature.append(sample)
            
    return synthetic_feature

def generate_synthetic_outcome(df, distribution):
    synthetic_outcome = []
    
    for _, row in df.iterrows():
        features = tuple(row[common_features])
        matching_row = distribution.loc[(distribution[common_features] == features).all(axis=1)]
        
        if not matching_row.empty:
            prob_yes = matching_row['y'].values[0]
            sample = np.random.choice([0, 1], p=[1 - prob_yes, prob_yes])
        else:
            sample = np.random.choice([0, 1])
        
        synthetic_outcome.append(sample)
    
    return synthetic_outcome


