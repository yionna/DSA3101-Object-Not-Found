
import os

if not os.path.exists('data_acquisition'):
    os.makedirs('data_acquisition')

# data_acquisition/fetch_economic_data.py
import pandas_datareader.data as web
import datetime
import pandas as pd

def fetch_economic_indicators(start_date, end_date):
    """
    Fetches economic indicators from FRED.
    Returns a DataFrame with the indicators.
    """
    indicators = {
        'GDP': 'GDP',                        
        'Unemployment_Rate': 'UNRATE',       
        'CPI': 'CPIAUCSL',                  
        'Consumer_Confidence': 'UMCSENT',   
    }
    
    data_frames = []
    for name, fred_code in indicators.items():
        df = web.DataReader(fred_code, 'fred', start_date, end_date)
        df.rename(columns={fred_code: name}, inplace=True)
        data_frames.append(df)
    
    # Merge all indicators on the date index
    economic_data = pd.concat(data_frames, axis=1)
    economic_data.reset_index(inplace=True)
    return economic_data

if __name__ == "__main__":
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime.now()
    economic_data = fetch_economic_indicators(start, end)
    
    # Save to CSV
    economic_data.to_csv('data/external/economic_indicators.csv', index=False)
    print("Economic indicators data fetched and saved.")
# data_acquisition/fetch_competitor_data.py
import pandas as pd
import numpy as np

def generate_competitor_data(segments):
    """
    Simulates competitor actions data for customer segments.
    Returns a DataFrame with competitor promotions and activities.
    """
    np.random.seed(42)
    data = {
        'Segment': segments,
        'Competitor_Promotion': np.random.choice([0, 1], size=len(segments)),
        'Competitor_Price_Change': np.random.choice([-1, 0, 1], size=len(segments)),
        'Competitor_New_Product': np.random.choice([0, 1], size=len(segments)),
    }
    competitor_data = pd.DataFrame(data)
    return competitor_data

if __name__ == "__main__":
    # Assume segments 1 to 5
    segments = [1, 2, 3, 4, 5]
    competitor_data = generate_competitor_data(segments)
    
    # Save to CSV
    competitor_data.to_csv('data/external/competitor_actions.csv', index=False)
    print("Competitor actions data generated and saved.")
# data_preprocessing.py
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype

# Load customer data
df = pd.read_csv('data/BankChurners_cleaned.csv')

# Load segmentation data
segmentation = pd.read_csv('data/segmentation_result_static.csv')

# Merge customer data with segmentation data
df1 = df.set_index('CLIENTNUM').join(segmentation.set_index('CLIENTNUM'), on='CLIENTNUM', how='inner').reset_index()

# Load external data
economic_data = pd.read_csv('data/external/economic_indicators.csv')
competitor_data = pd.read_csv('data/external/competitor_actions.csv')

# Convert 'Date' to datetime in economic_data
economic_data['DATE'] = pd.to_datetime(economic_data['DATE'])

# For simplicity, let's assume all customers are associated with the latest economic data
latest_economic_data = economic_data.sort_values('DATE').iloc[-1]
for col in ['GDP', 'Unemployment_Rate', 'CPI', 'Consumer_Confidence']:
    df1[col] = latest_economic_data[col]

# Merge competitor data on 'Segment'
df1 = df1.merge(competitor_data, on='Segment', how='left')

# Handle missing values if any
df1.fillna(method='ffill', inplace=True)

# Continue with your existing preprocessing steps...

# Define the levels for ordinal categorical variables
levels = {
    'Education_Level': ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
    'Income_Category': ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
    'Attrition_Flag': ['Existing Customer', 'Attrited Customer'],
    'Card_Category': ['Blue', 'Silver', 'Gold', 'Platinum'],
    'Gender': ['M', 'F'],
}

# Define nominal categorical variables
categorical = {
    'Marital_Status': ['Married', 'Single', 'Divorced'],
    'Competitor_Promotion': [0, 1],
    'Competitor_Price_Change': [-1, 0, 1],
    'Competitor_New_Product': [0, 1],
}

# Encode categorical variables
for col in df1.columns:
    if is_object_dtype(df1[col]):
        if col in levels:
            # Ordinal encoding
            df1[col] = pd.Categorical(df1[col], categories=levels[col], ordered=True)
            df1[col] = df1[col].cat.codes
        elif col in categorical:
            # Nominal encoding
            df1[col] = pd.Categorical(df1[col], categories=categorical[col])
            df1[col] = df1[col].cat.codes

# One-hot encode any remaining categorical variables
df1 = pd.get_dummies(df1, columns=['Marital_Status'], drop_first=True)

# Save the preprocessed data
df1.to_csv('data/preprocessed_data.csv', index=False)
print("Data preprocessing completed and saved.")
# modeling.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load preprocessed data
df2 = pd.read_csv('data/preprocessed_data.csv')

# Convert product columns into binary variables
product_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

for product in product_list:
    df2[product] = np.where(df2[product] > 0, 1, 0)

# Define customer clusters
clusters = {'general': [1, 4, 5], 'high_value': [2, 3]}

# Function to select best features
def select_best(x, y, k=10, score_func=f_regression):
    KBest = SelectKBest(score_func, k)
    KBest.fit(x, y)
    cols = KBest.get_support(indices=True)
    return x.columns[cols], pd.DataFrame({'features': x.columns, 'score': KBest.scores_, 'p-value': KBest.pvalues_})

# Initialize dictionaries
scores = {key: {} for key in clusters}
cols = {key: {} for key in clusters}
training_features = {key: {} for key in clusters}
predicted_labels = {key: {} for key in clusters}
actual_labels = {key: {} for key in clusters}

# Loop through clusters and products to train models
for key, cluster in clusters.items():
    users = df2[df2['Segment'].isin(cluster)]
    for product in product_list:
        # Prepare target variable
        y = users[product]
        
        # Prepare training features (exclude product columns and 'CLIENTNUM')
        X = users.drop(columns=product_list + ['CLIENTNUM'])
        
        # Feature selection
        selected_features, feature_scores = select_best(X, y, k=10)
        cols[key][product] = selected_features
        
        # Prepare training data
        X_selected = X[selected_features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.4, random_state=123
        )
        
        # Train XGBClassifier
        bst = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )
        bst.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = bst.predict_proba(X_test)[:, 1]
        predicted_labels[key][product] = pd.DataFrame(
            y_pred_proba, index=X_test.index, columns=[product]
        )
        actual_labels[key][product] = y_test

# Function to consolidate results
def get_results(results, source):
    products = list(results.keys())
    final_df = results[products[0]].copy()
    for product in products[1:]:
        final_df = final_df.join(results[product], how='left')
    final_df['Cluster'] = source
    return final_df.reset_index()

# Combine predictions from all clusters
predictions_list = []
for source in clusters:
    pldf = get_results(predicted_labels[source], source)
    predictions_list.append(pldf)
combined_predictions = pd.concat(predictions_list, ignore_index=True)

# Save combined predictions
combined_predictions.to_csv('data/predictions.csv', index=False)
print("Modeling completed and predictions saved.")
# Analyzing feature importance for one of the models
import matplotlib.pyplot as plt

# Example: For 'general' cluster and product 'A'
key = 'general'
product = 'A'

# Retrieve the model and feature names
model = bst  # The last trained model in the loop
feature_names = cols[key][product]

# Plot feature importance
xgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.title(f'Feature Importance for Product {product} in Cluster {key}')
plt.show()
