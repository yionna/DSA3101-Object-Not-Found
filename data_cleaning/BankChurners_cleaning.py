import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils import clean_col

##### This script will clean the BankChurners.csv file and save it as BandChuners_cleaning.csv in data/processed/
##### Note that the working directory is assumed to be where this file is found

# this function will remove the k,$ and + sign in the income category column of BankChurners.csv
def clean_col(x):
        if 'K' in x:
            return x.replace('K','').replace('$','')
        elif '+' in x:
            return x.replace('+','')
        elif x =='Less than 40':
            return x.split()[2]
        return x

## Start
original = pd.read_csv("../data/raw/BankChurners.csv")

# removing irrelevent columns
original = original.drop(original.columns[[-1, -2]], axis=1)

# removing Na from the dataset
original_Unknown = original[original.isin(['Unknown']).any(axis=1)] # someone handle the unknown please
original = original[~original.isin(['Unknown']).any(axis=1)]

original['Income_Category']=original['Income_Category'].apply(clean_col)

# Converting object into category
categorical_features = ['Attrition_Flag','Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
for category in categorical_features:
    original[category] = original[category].astype('category')


## End: saving as BankChurners_cleaned.csv
original.to_csv("../data/processed/BankChurners_cleaned.csv", index = False)
