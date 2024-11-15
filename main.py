import pandas as pd

from models.customerSegmentation.static import CustomerSegmentation
from models.RecommendationSystem.recommendationsystem import obtain_best_columns, model_train_test
from models.DynamicCampaign.DynamicCampaignSystem import get_campaign

# Importing data
base_df = pd.read_csv("data/processed/BankChurners_more.csv")
banking_df = pd.read_csv("data/processed/banking_behaviour_preference.csv")
recommendation_df = pd.read_csv("data/processed/recommendation_system_dataset.csv") # dataset with additional product data
demographic = pd.read_csv("data/processed/demographic.csv")

# Customer segmentation
segmentation = CustomerSegmentation(banking_df)
segmentation_result = segmentation.perform_segmentation()
print(segmentation_result['Segment'].value_counts())

# Recommendation system
clusters = {'low_income': [1, 3], 'medium_income': [2],'high_income': [4, 5]}
product_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
cols, scores = obtain_best_columns(recommendation_df, clusters, product_list)
predicted_labels, actual_labels = model_train_test(recommendation_df, cols, clusters, product_list)
print(predicted_labels)

# Campaign System
new_demographic = get_campaign(demographic)



