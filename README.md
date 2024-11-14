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

### Data Preparation
1) run `BankChurners_cleaning.ipynb` in `data_cleaning` to get `BankChurners_cleaned.csv` in `data/processed`

### Data Synthesis
*All the files are in `data_synthesis`*
1) run `BankChurner_more.ipynb` in `data_synthesis` to get `BankChurners_more.csv` in `data/processed`

#### For Subgroup A:
1) run `banking_behaviour_preference.ipynb` to get `banking_behaviour_preference.csv` in `data/processed`
2) run `Campaign_data.ipynb` to get `Campaign.csv` in `data/processed`

#### For Subgroup B:
1) 


### Group Tasks
#### Subgroup A:
*All the files are in `group_tasks/Subgroup A`*  
Qn 1: run `qn1.ipynb`  
Qn 2: run `qn2.ipynb`  
Qn 3: run `qn3.ipynb`

#### Additional Questions
Qn 1: run qn1(optional).ipynb
Qn 2: run qn2(optional).ipynb
Qn 3: run qn3(optional).ipynb 


#### Subgroup B:
*All the files are in `group_tasks/Subgroup B`*  
Qn 1: run `qn1.ipynb`  
Qn 2: run `qn2.ipynb`  
Qn 3: run `qn2.ipynb`

#### Additional Questions
Qn 1: run  
Qn 2: run  
Qn 3: run  

## Repository structure
```

├── data_cleaning                            # Cleaning the original dataset
│   ├── BankChurners_cleaning.py
├──                       # Exploratory data analysis
├── data_synthesis                           # Adding rows and columns to main data
│   ├── BankChurners_more.ipynb              # Synthesise more rows into `BankChurners_cleaned.csv`
│   ├── banking_behaviour_preference.ipynb   
│   ├── Campaign_data.ipynb
│   └── Product_data.ipynb
├── data_engineering
│   ├── campaign_log.py
├── data/                          
│   ├── predictions                          # For API
│   │   ├── A_BQ3.csv                        # For Churn Likelihood
│   │   ├── A_BQ3_pt2.csv                    # For risk of churning
│   │   ├── BQ1.csv
│   │   └── segementation_result_static.csv
│   ├── processed                            # Processed Data
│   │   ├── BankChurners_cleaned.csv
│   │   ├── BankChurners_more.csv
│   │   ├── Campaign.csv
│   │   ├── Compaign_metrics.csv
│   │   ├── banking_behaviour_preference.csv
│   │   ├── banking_behaviour_preference_original.csv
│   │   ├── income_category_mapping.json
│   │   ├── product_services_data.csv
│   │   ├── recommendation_system_dataset.csv
│   │   ├── segmentation_result_static.csv
│   │   ├── segmentation_result_static_original.csv
│   │   └── simulation_result.csv
│   ├── raw                                  # Original Datasets
│   │   ├── BankChurners.csv
│   │   ├── User churn.csv
│   │   ├── bank_reviews3.csv
│   │   ├── banking_product_services.csv
│   │   ├── botswana_bank_customer_churn.csv
│   │   ├── campaign_data.csv
│   │   └── credit_score.csv
│   └── subgroupA_visuals                    # Visualisations for Subgroup A Question 3
│       ├── output1.png
│       ├── output2.png
│       ├── output3.png
│       ├── output4.png
│       ├── output5.png
│       └── output6.png
├── sql                                      # SQL scripts for data extraction
and transformation
│   └── setup_bank_db.sql
├── API                                      # Simple API (using Flask or FastAPI) to serve model predictions and key insights
│   ├── static
│   │   ├── styles_customer_info.css
│   │   └── styles_index.css
│   ├── templates
│   │   ├── customer_information.html
│   │   └── index.html
│   ├── app.py
│   └── dashboard.py 
├── requirements.txt                         # All dependencies
├── models                                   # For all functions, classes, and modules
│   ├── AIMarketing
│   │   └── marketing.py
│   ├── CampaignUpdate
│   │   └── UpdateCampaign.py
│   ├── DynamicCampaign
│   │   └── DynamicCampaignSystem.py
│   ├── EarlyWarningSystem
│   │   └── EarlyWarningSystem.py
│   ├── RecommendationSystem
│   │   └── recommendationsystem.py
│   └── customerSegmentation
│       ├── CustomerDataGenerator.py
│       ├── dynamic.py
│       └── static.py
├──                                          # Dockerfile to containerize the application
├── group tasks/
│   ├── Subgroup A
│   │   ├── qn1.ipynb
│   │   ├── qn2.ipynb
│   │   ├── qn3.ipynb
│   │   ├── qn1 (optional).ipynb
│   │   ├── qn2 (optional).ipynb
│   │   └── qn3 (optional).ipynb
│   └── Subgroup B
│       ├── qn1.ipynb
│       ├── qn2.ipynb
│       ├── qn3.ipynb
│       ├── extra_qn2.ipynb
│       └──
├── main.py                                  # Orchestrates the entire data pipeline and analysis process
├── config.py                                # For all configuration parameters
├── utils.py                                 # For utility functions used across multiple scripts
└── README.md                                # Project documentation
```

## Data sources and any necessary data preparation steps
### Data sources:
- `data/BankChurners.csv`
- `data/botswana_bank_customer_churn.csv`
- `data/User churn.csv`
  link: https://www.kaggle.com/datasets/mikhail1681/user-churn<br>
  To get the variables that are related to digital engagements.<br>
- `data/credit_score.csv`<br>
  link: https://www.kaggle.com/datasets/conorsully1/credit-score?resource=download<br>
  To get the `Savings` variable that includes savings outside the bank.<br>
- `data/campaign_data.csv`<br>
  link: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=test.csv<br>
  The training data is used since it was much larger than the test data (randomly selected rows from training data).<br>
- `data/raw/bank_reviews3.csv`<br>
  link: https://www.kaggle.com/datasets/dhavalrupapara/banks-customer-reviews-dataset/data<br>
  The data is used as a sample dataset for showcasing the functions of the NLP pipeline.<br>

## Instructions for building and running the Docker containers

## API documentation
### Data:
* From `group_tasks/Subgroup A/qn3(optional).ipynb`:
    * for customer churn likelihood: `data/predictions/A_BQ3.csv`
    * for risk of churning: `data/predictions/A_BQ3_pt2.csv`

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
