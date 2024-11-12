import pandas as pd
from models.customerSegmentation.static import *

# Importing data
base_df = pd.read_csv("data/processed/BankChurners_more.csv")
banking_df = pd.read_csv("data/processed/banking_behaviour_preference.csv")

# Customer segmentation
segmentation_test = CustomerSegmentation(banking_df)
segmentation_result = segmentation_test.perform_segmentation()