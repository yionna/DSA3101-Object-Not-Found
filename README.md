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

1) run `data_cleaning.ipynb` using `BankChurners.csv` (our original dataset) to get `BankChurners_cleaned.csv`
2) run 'data_analysis.ipynb` to get the exploratory data analysis of the original cleaned dataset

#### For Subgroup A:
##### Data synthesis
1) run `botswana_bank.ipynb` in `data synthesis/botswana_bank.ipynb` using `botswana_bank_customer_churn.csv` to add `Credit Score`, `Outstanding Loans` and `Balance` to the original cleaned dataset --> `temp4.csv`
2) run `User_churn.ipynb` in `data synthesis/User_churn.ipynb` using `User churn.csv` to generate data for digital engagement, then combining this data with `temp4.csv` --> `original.csv`
3) run `Credit_Score.ipynb` in `data synthesis/Credit_Score.ipynb` using `credit_score.csv` to add `Savings` (savings outside the bank) to the original cleaned dataset --> `original (2).csv`
4) run `Campaign_data.ipynb` in `data synthesis/Campaign_data.ipynb` using `campaign_data.csv` to add `Duration_of_Contact`, `Number_of_Contacts_Made`, `Last_Contact_Made` and `Outcome` to `original (2).csv` --> `original (3).csv`

#### Questions
Question 1: run `qn1.ipynb` in `group tasks/Subgroup A/qn1.ipynb` using `original.csv` to get `original (4).csv`<br>
Question 2: run `qn2.ipynb` in `group tasks/Subgroup A/qn2.ipynb` where `original (3).csv` and `original (4).csv` merge to get `original (5).csv`<br>
Question 3: <br>

Bonus tasks:<br>
Question 2:<br>
Question 3: run 'qn3 (bonus).ipynb` in `group tasks/Subgroup A/qn3 (bonus).ipynb` using `original (5).csv`<br>

### Environment setup

### Running the code

## Repository structure
```
personalized-marketing-bank/
├── data_cleaning.ipynb                      # Cleaning the original dataset
├── data_analysis.ipynb                      # Exploratory data analysis
├── data synthesis/                          # Adding columns to data
│   ├── botswana_bank.ipynb                  
│   ├── User_churn.ipynb
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

### Data Dictionary

#### 1) Data Dictionary from original Dataset
| Variable               | Data Type   | Description                                            | Example Value        |
|------------------------|-------------|--------------------------------------------------------|-----------------------|
| CLIENTNUM              | Integer     | Unique identifier for each customer                    | 768805383            |
| Attrition_Flag         | String      | Indicates if the customer has churned or not           | Existing Customer    |
| Customer_Age           | Integer     | Age of the customer                                   | 45                   |
| Gender                 | String      | Gender of the customer                                 | M                    |
| Dependent_count        | Integer     | Number of dependents associated with the customer      | 3                    |
| Education_Level        | Categorical | Education level of the customer                        | Graduate             |
| Marital_Status         | Categorical | Marital status of the customer                         | Married              |
| Income_Category        | Categorical | Income bracket/category of the customer                | $60K - $80K          |
| Card_Category          | Categorical | Type/category of card held by the customer             | Blue                 |
| Month_with_bank        | Integer     | Number of months the customer has been with the bank   | 24                   |
| No_of_product          | Integer     | Number of products the customer is using               | 2                    |
| Months_Inactive_12_mon | Integer     | Number of months customer was inactive in the last 12 months | 3            |
| Credit_Limit           | Float       | Credit limit for the customer                          | 15000.0              |
| Total_Revolving_Bal    | Float       | Total revolving balance for the customer               | 500.0                |
| Total_Trans_Amt        | Float       | Total transaction amount                               | 4000.0               |
| Total_Trans_Count      | Integer     | Total count of transactions                            | 50                   |
| Avg_Utilization_Ratio  | Float       | Average credit card utilization ratio                  | 0.2                  |
| Savings                | Float       | Savings balance of the customer                        | 3000.0               |
| Duration_of_Contact    | Integer     | Duration (in minutes) of the last contact              | 15                   |
| Num_of_Contacts_Made   | Integer     | Number of contacts made with the customer              | 5                    |
| Last_Contacted         | Date        | Date of the last contact with the customer             | 2024-01-15           |
| Outcome                | String      | Outcome of the last interaction                        | Successful           |
| Credit Score           | Integer     | Credit score of the customer                           | 750                  |
| Outstanding Loans      | Float       | Total amount of outstanding loans                      | 20000.0              |
| Balance                | Float       | Account balance                                        | 5000.0               |
| PhoneService           | Boolean     | Indicates if customer has phone service                | Yes                  |
| InternetService        | Boolean     | Indicates if customer has internet service             | No                   |
| TechSupport            | Boolean     | Indicates if customer has tech support                 | Yes                  |
| PaperlessBilling       | Boolean     | Indicates if customer has opted for paperless billing  | No                   |
| PaymentMethod          | Categorical | Preferred payment method                               | Credit Card          |
| Digital_Engagement     | Integer     | Digital engagement score based on interactions         | 85                   |
| Financial_Status       | Categorical | Financial status category                              | Stable               |
| Transaction_Behavior   | Float       | Behavior score based on transaction patterns           | 1.5                  |
| Product_Usage          | Float       | Score representing product usage frequency             | 2.3                  |
| Banking_Behavior       | Float       | Banking behavior score                                 | 3.0                  |
| Customer_Preferences   | Float       | Score representing customer preferences                | 7.5                  |
| Cluster_Labels         | Categorical | Cluster label assigned based on behavioral analysis    | High Engagement      |
