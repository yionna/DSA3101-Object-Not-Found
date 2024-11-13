import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import json

##### This script will clean the BankChurners.csv file and save it as BandChuners_cleaning.csv in data/processed/
##### Note that the working directory is assumed to be where this file is found.
##### Some label encoding is not done in this section because different data synthesis requires different way to encode the values.

# this function will remove the k,$ and + sign in the income category column of BankChurners.csv
def clean_col(x):
        if 'K' in x:
            return x.replace('K','').replace('$','')
        elif '+' in x:
            return x.replace('+','')
        elif x =='Less than 40':
            return x.split()[2]
        return x

# Function to encode 'Attrition_Flag' column
def encode_card(flag):
    if flag == "Blue":
        return 1
    elif flag == "Silver":
        return 2
    else:
        return 3

## Start
original = pd.read_csv("../data/raw/BankChurners.csv")

# Removing irrelevent columns
original = original.drop(original.columns[[-1, -2]], axis=1)
original = original.drop(columns=['Total_Amt_Chng_Q4_Q1','Contacts_Count_12_mon','Total_Ct_Chng_Q4_Q1'])

# Renaming 'Total_relationship_count' to 'No_of_product' for easier understanding
original = original.rename(columns={'Months_on_book' : 'Month_with_bank',
                                    'Total_Relationship_Count' : 'No_of_product',
                                    'Total_Trans_Ct' : 'Total_Trans_Count'})

# Filter out rows where 'Card_Category' is 'Other'(<1%)
original = original.loc[original['Card_Category'] != 'Other']
# Encoding 'Card_Category'
original['Card_Category'] = original['Card_Category'].apply(encode_card)
# Encoding 'Gender'
original['Gender'] = original['Gender'].replace({'F': 0, 'M': 1})
# Encode 'Attrition_Flag' in-place, setting 'Existing Customer' to 1 and others to 0
original['Attrition_Flag'] = original['Attrition_Flag'].replace({'Existing Customer': 1, 'Attrited Customer': 0})
# Encoding 'Marital_Status'
original['Marital_Status'] = original['Marital_Status'].replace({'Divorced': 0, 'Married': 1, 'Single': 0})


# Splitting Na(Unknown) from the dataset
original_Unknown = original[original.isin(['Unknown']).any(axis=1)]
original = original[~original.isin(['Unknown']).any(axis=1)]

original['Income_Category']= original['Income_Category'].apply(clean_col)
# Encode 'Income_category'
le_income = LabelEncoder()
original['Income_Category'] = le_income.fit_transform(original['Income_Category'])

# Create the mapping dictionary and convert values to Python int
income_category_mapping = {key: int(value) for key, value in zip(le_income.classes_, le_income.transform(le_income.classes_))}

# Save the mapping to a JSON file
with open("../data/processed/income_category_mapping.json", "w") as file:
    json.dump(income_category_mapping, file)


# Converting object into category
categorical_features = ['Attrition_Flag','Gender','Education_Level','Marital_Status','Income_Category']
for category in categorical_features:
    original[category] = original[category].astype('category')

## Handling Na's(Unknown)
unknown_columns = original_Unknown.isin(['Unknown']).sum()
print(unknown_columns)

# We can see that only 3 columns contain Na which are Education_Level,Marital_Status,Income_Category.
# Hence , we will try to impute value for these columns

## Data imputation
# We will make use of a decision tree to do the imputation

# Define columns with missing values
missing_columns = ['Education_Level', 'Marital_Status', 'Income_Category']

for column in missing_columns:
    X_train = original.drop(columns=missing_columns)
    y_train = original[column]
    
    # Select rows in original_Unknown for the imputation set
    X_missing = original_Unknown.drop(columns=missing_columns)
    
    # Initialize and train the DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Predict missing values and assign them back to original_Unknown
    original_Unknown[column] = model.predict(X_missing)
    

# Verify that 'Unknown' values have been removed from original_Unknown
if not original_Unknown.isin(['Unknown']).any().any():
    print("No 'Unknown' values found in original_Unknown. Ready to merge.")

    # Concatenate original and original_Unknown back together
    final_data = pd.concat([original, original_Unknown])

    # Save the final cleaned dataset
    final_data.to_csv("../data/processed/BankChurners_cleaned.csv", index=False)
    print("Data imputation complete and saved as 'BankChurners_cleaned.csv'")
else:
    print("There are still 'Unknown' values in original_Unknown. Please review the imputation process.")


## End