# DSA3101-Object-Not-Found: Machine Learning for Personalised Marketing Campaigns

## Project Overview

pts we could include:
- what the marketing specialists have "told" us
- problems with marketing without ML/AI
- what we did
- what we will be telling the IT specialists to do from this project (maybe if it flows smoothly)


Marketing specialists on our team have expressed the need for a more targeted campaign since the current more "traditional" marketing methods rely mostly on broad demographic segments and basic metrics, not fully utilising the wealth of available customer data. This causes campaigns not to consider more nuanced features like customer preferences, behaviours, and needs.

### Problems with Marketing without Machine Learning (ML)/ Artificial Intelligence (AI)
There will be a lack of personalisation as customers will receive generic, one-size-fits-all offers or recommendations that do not align with the customers' preferences. There is also resource inefficiency since such campaigns lead to inefficient allocation of resources as it will be difficult to identify customers who are most likely to engage.

## Setting up the environment and running the code
### Requirements:
- Python 3.8+
- Docker
- Other dependencies listed in ´requirements.txt´

### Data preparation
1) run `data_cleaning.ipynb` using `BankChurners.csv` (our original dataset) to get `BankChurners_cleaned.csv`
2) run 'data_analysis.ipynb` to get the exploratory data analysis of the original cleaned dataset

### For Subgroup A:
#### Data synthesis
1) run `banking_behaviour_preference.ipynb` in `data synthesis/banking_behaviour_preference.ipynb` to get `processed/banking_behaviour_preference.csv`
2) run `Credit_Score.ipynb` in `data synthesis/Credit_Score.ipynb` using `credit_score.csv` to add `Savings` (savings outside the bank) to the original cleaned dataset --> `original (2).csv`
3) run `Campaign_data.ipynb` in `data synthesis/Campaign_data.ipynb` using `campaign_data.csv` to add `Duration_of_Contact`, `Number_of_Contacts_Made`, `Last_Contact_Made` and `Outcome` to `original (2).csv` --> `original (3).csv`

#### Questions
Question 1: run `qn1.ipynb` in `group tasks/Subgroup A/qn1.ipynb` using `processed/banking_behaviour_preference.csv` to get `segementation_result_static.csv`<br>
Question 2: run `qn2.ipynb` in `group tasks/Subgroup A/qn2.ipynb` where `original (3).csv` and `banking_behaviour_preference.csv` merge to get `original (5).csv`<br>
Question 3: <br>

Bonus tasks:<br>
Question 2:<br>
Question 3: run 'qn3 (bonus).ipynb` in `group tasks/Subgroup A/qn3 (bonus).ipynb` using `original (5).csv`<br>


## Repository structure
```
personalized-marketing-bank/
├── data_cleaning.ipynb                      # Cleaning the original dataset
├── data_analysis.ipynb                      # Exploratory data analysis
├── data synthesis/                          # Adding columns to data
│   ├── banking_behaviour_preference.ipynb
│   ├── Credit_score.ipynb
│   ├── Campaign_data.ipynb
│   └──
├── main.py                                  # Orchestrates the entire data pipeline and analysis process
├── config.py                                # For all configuration parameters
├── utils.py                                 # For utility functions used across multiple scripts
├──                                          # SQL scripts for data extraction and transformation
├──                                          # Simple API (using Flask or FastAPI) to serve model predictions and key insights
├── requirements.txt                         # All dependencies
├──                                          # For all functions, classes, and modules
├──                                          # Dockerfile to containerize the application
├── data/                                    # Contains raw and preprocessed data files
│   ├── BankChurners.csv                     # Main dataset
│   ├── botswana_bank_customer_churn.csv     # For `Credit Score`, `Outstanding Loans` and `Balance`
│   ├── User churn.csv                       # For data for digital engagement
│   ├── credit_score.csv                     # For `Savings`
│   ├── campaign_data.csv                    # For `Duration_of_Contact`, `Number_of_Contacts_Made`, `Last_Contact_Made` and `Outcome`
│   └── 
├── group tasks/
│   ├── Subgroup A
│   │   ├── qn1.ipynb
│   │   ├── qn2.ipynb
│   │   ├──
│   │   ├──
│   │   └── qn3 (bonus).ipynb
│   └── Subgroup B
│   │   ├── 
│   │   ├── 
│   │   ├──
│   │   ├──
│   │   └── 
└── README.md                                # Project documentation
```

## Data sources and any necessary data preparation steps
### Data sources:
- `data/BankChurners.csv`
- `data/botswana_bank_customer_churn.csv`
- `data/User churn.csv`
- `data/credit_score.csv`<br>
  link: https://www.kaggle.com/datasets/conorsully1/credit-score?resource=download<br>
  To get the ´Savings´variable that includes savings outside the bank<br>
- `data/campaign_data.csv`<br>
  link: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=test.csv<br>
  The training data is used since it was much larger than the test data (randomly selected rows from training   data).

## Instructions for building and running the Docker containers

## API documentation

### Endpoints

### Request/Response formats

## Data Dictionary

### 1) Data Dictionary for original Dataset
| Variable               | Data Type   | Description                                                             | Example Value |
|------------------------|-------------|-------------------------------------------------------------------------|---------------|
| CLIENTNUM              | Integer     | Unique identifier for each customer                                     | 768805383     |
| Customer_Age           | Integer     | Age of the customer                                                     | 45            |
| Gender                 | Categorical | Gender of the customer (1 = Male, 0 = Female)                           | 1             |
| Dependent_count        | Integer     | Number of dependents                                                    | 3             |
| Education_Level        | Categorical | Education level of the customer (e.g., 1 = High School, 2 = Graduate)   | 3             |
| Marital_Status         | Categorical | Marital status of the customer (e.g., 1 = Married, 2 = Single)          | 1             |
| Income_Category        | Categorical | Income bracket of the customer (e.g., 2 = 60 - 80, 4 = Less than 40)    | 2             |
| Card_Category          | Categorical | Type of credit card held by the customer                                | 0             |
| Month_with_bank        | Integer     | Number of months the customer has been with the bank                    | 39            |
| No_of_product          | Integer     | Number of products held by the customer                                 | 5             |
| Savings                | Float       | Total savings balance of the customer                                   | 452014.73     |
| Duration_of_Contact    | Integer     | Duration of the customer’s contact with the bank (in months)            | 73            |
| Num_of_Contacts_Made   | Integer     | Number of contacts made by the customer                                 | 2             |
| Last_Contacted         | Integer     | Last contacted time (-1 indicates not contacted recently)               | -1            |
| Outcome                | Integer     | Outcome of recent bank interactions (e.g., 0 = Neutral, 1 = Positive)   | 0             |
| Loyalty                | Integer     | Loyalty score of the customer                                           | 4             |
| Financial_Status       | Integer     | Financial status rating of the customer                                 | 7             |
| Segment                | Categorical | Segment classification of the customer (e.g., 1 = Standard, 2 = Premium)| 2             |
| Digital_Capability     | Boolean     | Digital engagement status (True = Engaged, False = Not Engaged)         | True          |
| Churn                  | Integer     | Churn indicator (0 = No churn, 1 = Churned)                             | 0             |


### 2) Data Dictionary for banking_behaviour_preference Dataset
| Variable              | Data Type   | Description                                                | Example Value      |
|-----------------------|-------------|------------------------------------------------------------|--------------------|
| CLIENTNUM             | Integer     | Unique identifier for each customer                        | 768805383          |
| Income_Category       | Categorical | Income bracket of the customer (e.g. 4 = Less than 40)     | 2                  |
| No_of_product         | Integer     | Number of products the customer is using                   | 5                  |
| Total_Trans_Amt       | Float       | Total transaction amount for the customer                  | 1144               |
| Total_Trans_Count     | Integer     | Total count of transactions made by the customer           | 42                 |
| Credit Score          | Float       | Credit score of the customer                               | 623.80             |
| Outstanding Loans     | Float       | Total amount of outstanding loans                          | 13384.62           |
| Balance               | Float       | Current account balance of the customer                    | 159276.45          |
| PhoneService          | Integer     | Indicates if the customer has phone service (1 = Yes, 0 = No) | 1               |
| InternetService       | Integer     | Indicates if the customer has internet service (1 = Yes, 0 = No) | 0             |
| TechSupport           | Integer     | Indicates if the customer has tech support (2 = High, 1 = Moderate, 0 = None) | 2         |
| PaperlessBilling      | Integer     | Indicates if the customer has opted for paperless billing (1 = Yes, 0 = No) | 1         |
| PaymentMethod         | Categorical | Payment method used by the customer                        | 2 (Credit Card)    |


### 3) Data Dictionary for BankChurners_cleaned Dataset
| Variable               | Data Type   | Description                                                              | Example Value      |
|------------------------|-------------|--------------------------------------------------------------------------|---------------------|
| CLIENTNUM              | Integer     | Unique identifier for each customer                                      | 768805383          |
| Attrition_Flag         | Categorical | Indicates if the customer is existing or attrited (e.g. 1 = Existing)    | Existing Customer  |
| Customer_Age           | Integer     | Age of the customer                                                      | 45                 |
| Gender                 | Categorical | Gender of the customer (e.g. 1 = Male)                                   | 1                  |
| Dependent_count        | Integer     | Number of dependents the customer has                                    | 3                  |
| Education_Level        | Categorical | Highest education level achieved by the customer                         | High School        |
| Marital_Status         | Categorical | Marital status of the customer (e.g., 1 = Married, 0 = Single)           | 1                  |
| Income_Category        | Categorical | Income bracket of the customer (e.g., 2 = 60 - 80, 4 = Less than 40)     | 2                  |
| Card_Category          | Categorical | Type of credit card held by the customer (e.g. 2 = gold card)            | 1                  |
| Month_with_bank        | Integer     | Number of months the customer has been with the bank                     | 39                 |
| No_of_product          | Integer     | Number of products the customer is using                                 | 5                  |
| Months_Inactive_12_mon | Integer     | Number of months the customer was inactive in the last 12 months         | 1                  |
| Credit_Limit           | Float       | Credit limit assigned to the customer                                    | 12691.0            |
| Total_Revolving_Bal    | Float       | Total revolving balance on the account                                   | 777                |
| Total_Trans_Amt        | Float       | Total transaction amount for the customer                                | 1144               |
| Total_Trans_Count      | Integer     | Total count of transactions made by the customer                         | 42                 |
| Avg_Utilization_Ratio  | Float       | Average credit utilization ratio                                         | 0.061              |

### 4) Data Dictionary for digital_marketing_campaign_dataset Dataset
| Variable              | Data Type   | Description                                                       | Example Value           |
|-----------------------|-------------|-------------------------------------------------------------------|--------------------------|
| CustomerID            | Integer     | Unique identifier for each customer                               | 8000                     |
| Age                   | Integer     | Age of the customer                                               | 56                       |
| Gender                | Categorical | Gender of the customer                                            | Female                   |
| Income                | Float       | Annual income of the customer                                     | 136912                   |
| CampaignChannel       | Categorical | The channel through which the campaign was delivered              | Social Media             |
| CampaignType          | Categorical | Type or purpose of the campaign                                   | Awareness                |
| AdSpend               | Float       | Amount spent on advertisements for the campaign                   | 6497.87                  |
| ClickThroughRate      | Float       | Ratio of clicks to impressions in the campaign                    | 0.0439                   |
| ConversionRate        | Float       | Percentage of customers who completed the desired action          | 0.088                    |
| WebsiteVisits         | Integer     | Number of visits to the website generated by the campaign         | 0                        |
| PagesPerVisit         | Float       | Average number of pages viewed per visit                          | 2.399                    |
| TimeOnSite            | Float       | Average time spent on the site (in minutes)                       | 7.397                    |
| SocialShares          | Integer     | Number of times the campaign content was shared on social media   | 19                       |
| EmailOpens            | Integer     | Number of times the campaign email was opened                     | 6                        |
| EmailClicks           | Integer     | Number of clicks from campaign emails                             | 9                        |
| PreviousPurchases     | Integer     | Number of purchases the customer has made prior to this campaign  | 4                        |
| LoyaltyPoints         | Integer     | Loyalty points accumulated by the customer                        | 688                      |
| AdvertisingPlatform   | Categorical | Platform used for campaign advertisements                         | IsConfid                 |
| AdvertisingTool       | Categorical | Tool or technology used for advertising                           | ToolConfid               |
| Conversion            | Integer     | Indicator if the campaign led to a conversion (1 = Yes, 0 = No)   | 1                        |
