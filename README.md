# DSA3101-Object-Not-Found: Machine Learning for Personalised Marketing Campaigns

As clients become more digitally savvy, marketing efforts in the banking industry have shifted away from traditional methods like mass media marketing to digital marketing. Current digital marketing efforts rely mostly on broad demographic segments and basic metrics, underutilising available consumer data. Our team’s marketing specialists have thus asked for our expertise in identifying the relevant segments to target their marketing campaign and come up with strategies to address low engagement with customers. Our team and data scientists have thus proposed creating more personalised marketing strategies that are tailored to customers’ preferences, behaviours and needs. Overall, our project aims to improve customer engagement, increase conversion rates, and enable data-driven decision-making in marketing strategies.

## Project Overview
Marketing specialists on our team have expressed the need for a more targeted campaign since the current more "traditional" marketing methods rely mostly on broad demographic segments and basic metrics, not fully utilising the wealth of available customer data. This causes campaigns not to consider more nuanced features like customer preferences, behaviours, and needs.

### Problems with Marketing without Machine Learning (ML)/ Artificial Intelligence (AI)
There will be a lack of personalisation as customers will receive generic, one-size-fits-all offers or recommendations that do not align with the customers' preferences. There is also resource inefficiency since such campaigns lead to inefficient allocation of resources as it will be difficult to identify customers who are most likely to engage.

## Setting up the environment and running the code
### Requirements:
- Python 3.8+
- Docker
- Other dependencies listed in ´requirements.txt´
- `libomp` (required for XGBoost on macOS)

### Data Preparation
1) run `BankChurners_cleaning.ipynb` in `data_cleaning` to get `BankChurners_cleaned.csv` in `data/processed`

### Data Synthesis
*All the files are in `data_synthesis`*
1) run `BankChurner_more.ipynb` in `data_synthesis` to get `BankChurners_more.csv` in `data/processed`

#### For Subgroup A:
1) run `banking_behaviour_preference.ipynb` to get `banking_behaviour_preference.csv` in `data/processed`
2) run `Campaign_data.ipynb` to get `Campaign.csv` in `data/processed`

#### For Subgroup B:
1) run `Product_data` to get `recommendation_system_data.csv` in `data/processed`
2) run `Demographic_data_without_result.ipynb` to get `demographic.csv` in `data/processed`

### Group Tasks
#### Subgroup A:
*All the files are in `group_tasks/Subgroup A`*  
Qn 1: run `qn1.ipynb`  
Qn 2: run `qn2.ipynb`  
Qn 3: run `qn3.ipynb`

##### Additional Questions
Qn 1: run `qn1(optional).ipynb`  
Qn 2: run `qn2(optional).ipynb`  
Qn 3: run `qn3(optional).ipynb` 

#### Subgroup B:
*All the files are in `group_tasks/Subgroup B`*  
Qn 1: run `qn1.ipynb`  
Qn 2: run `qn2.ipynb`  
Qn 3: run `qn2.ipynb`

##### Additional Questions
Qn 2: run  `extra_qn2.ipynb`  
Qn 3: run  `extra_qn3_with_click.py`

## Repository structure
```

├── data_cleaning                            # Cleaning the original dataset
│   └── BankChurners_cleaning.py
├── data_synthesis                           # Adding rows and columns to main data
│   ├── BankChurners_more.ipynb              # Synthesise more rows into `BankChurners_cleaned.csv`
│   ├── banking_behaviour_preference.ipynb   
│   ├── Campaign_data.ipynb
│   ├── Demographic_data_without_result.ipynb
│   └── Product_data.ipynb
├── data
│   ├── Dashboard
│   │   ├── Dashboard_1_KPI.pbix
│   │   ├── Dashboard_1_KPI.png
│   │   ├── Dashboard_2_Tracking_Traffic.pbix
│   │   ├── Dashboard_2_Tracking_Traffic.pdf
│   │   └── dashboard.py                     # A demonstration of AI marketing
│   ├── predictions                          # For API
│   │   ├── A_BQ3.csv                        # For Churn Likelihood
│   │   ├── A_BQ3_pt2.csv                    # For risk of churning
│   │   ├── BQ1.csv                          # For product recommendation
│   │   ├── BQ2.csv                          # For Dynamic Campaign
│   │   └── segementation_result_static.csv
│   ├── processed                            # Processed Data
│   │   ├── BankChurners_cleaned.csv
│   │   ├── BankChurners_more.csv
│   │   ├── Campaign.csv
│   │   ├── Compaign_metrics.csv
│   │   ├── demographic.csv
│   │   ├── banking_behaviour_preference.csv
│   │   ├── banking_behaviour_preference_original.csv
│   │   ├── income_category_mapping.json
│   │   ├── product_services_data.csv
│   │   ├── recommendation_system_dataset.csv
│   │   ├── segmentation_result_static.csv
│   │   ├── segmentation_result_static_original.csv
│   │   └── simulation_result.csv
│   └── raw                                  # Original Datasets
│       ├── BankChurners.csv
│       ├── User churn.csv
│       ├── bank_reviews3.csv
│       ├── banking_product_services.csv
│       ├── botswana_bank_customer_churn.csv
│       ├── campaign_data.csv
│       ├── credit_score.csv
│       └── digital_marketing_campaign_dataset.csv
├── sql                                      # SQL scripts for data extraction and transformation
│   ├── setup_bank_db.sql
│   └── relational_data_bases.png            # Image of our relational database.
├── API                                      # Simple API (using Flask or FastAPI) to serve model predictions and key insights
│   ├── static
│   │   ├── styles_customer_info.css
│   │   ├── styles_display.css
│   │   ├── styles_index.css
│   │   └── subgroupA_visuals                    # Visualisations for Subgroup A Question 3
│   │       ├── output1.png
│   │       ├── output2.png
│   │       ├── output3.png
│   │       ├── output4.png
│   │       ├── output5.png
│   │       └── output6.png
│   ├── templates
│   │   ├── customer_information.html
│   │   └── index.html
│   ├── app.py
│   └── dashboard.py
├── models                                   # For all functions, classes, and modules
│   ├── AIMarketing
│   │   └── marketing.py
│   ├── FeedbackAnalysis
│   │   └── FeedbackAnalysis.py
│   ├── DynamicCampaign
│   │   └── DynamicCampaignSystem.py
│   ├── RecommendationSystem
│   │   └── recommendationsystem.py
│   ├── FeedbackAnalysis
│   │   └── FeedbackAnalysis.py
│   └── customerSegmentation
│       ├── CustomerDataGenerator.py
│       ├── dynamic.py
│       └── static.py
├── group_tasks
│   ├── Subgroup A
│   │   ├── qn1.ipynb
│   │   ├── qn2.ipynb
│   │   ├── qn3.ipynb
│   │   ├── qn1 (optional).ipynb
│   │   ├── qn2 (optional).ipynb
│   │   └── qn3 (optional).ipynb
│   └── Subgroup B
│       ├── extra_qn1.py
│       ├── extra_qn2.ipynb
│       ├── extra_qn2.py
│       ├── extra_qn3.py
│       ├── extra_qn3_with_click.py
│       ├── qn1.ipynb
│       ├── qn2.ipynb
│       └── qn3.ipynb
├── main.py                                  # Orchestrates the entire data pipeline and analysis process
├── Exploratory_Data_Analysis.ipynb          # Exploratory data analysis
├── config.py                                # For all configuration parameters
├── utils.py                                 # For utility functions used across multiple scripts
├── dockerfile                               # Defines the Docker image setup using a Conda base image, installing dependencies from `environment.yml`
├── environment.yml                          # Lists required packages for a reproducible Conda environment within Docker
├── requirements.txt                         # All dependencies
└── README.md                                # Project documentation
```

## Data sources and any necessary data preparation steps
### Data sources:
- `data/raw/BankChurners.csv`  
  link: https://www.kaggle.com/datasets/imanemag/bankchurnerscsv  
  Main dataset  
- `data/raw/botswana_bank_customer_churn.csv`  
  link: https://www.kaggle.com/datasets/sandiledesmondmfazi/bank-customer-churn  
  To get digital engagement data  
- `data/raw/User churn.csv`  
  link: https://www.kaggle.com/datasets/mikhail1681/user-churn<br>
  To get the variables that are related to digital engagements.<br>
- `data/raw/credit_score.csv`<br>
  link: https://www.kaggle.com/datasets/conorsully1/credit-score?resource=download<br>
  To get the `Savings` variable that includes savings outside the bank.<br>
- `data/raw/campaign_data.csv`<br>
  link: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets?select=test.csv<br>
  The training data is used since it was much larger than the test data (randomly selected rows from training data).<br>
- `data/raw/bank_reviews3.csv`<br>
  link: https://www.kaggle.com/datasets/dhavalrupapara/banks-customer-reviews-dataset/data<br>
  The data is used as a sample dataset for showcasing the functions of the NLP pipeline.<br>
- `data/raw/banking_product_services.csv`<br>
  link: https://www.kaggle.com/datasets/akhilups/insurance-product-purchase-prediction?resource=download<br>
  The data is used to obtain banking product information.<br>
- `data/raw/digital_marketing_campaign_dataset.csv`<br>
  link: https://www.kaggle.com/datasets/akhilups/insurance-product-purchase-prediction?resource=download<br>
  The data is used to obtain digital marketing campaign information.<br>

## Instructions for building and running the Docker containers
1. **Build the Docker Image**

   From within the project directory, build the Docker image using the following command:

   ```bash
   docker build -t dsa3101:1.0 .
   ```
2. **Run the Docker Container**

   Once the image is built, you can run the Docker container with an interactive session:

   ```bash
   docker run -it dsa3101:1.0 /bin/bash
   ```
   This command will start an interactive session within the container, allowing you to run code and access the environment directly.
3. **Activate the Conda Environment**

   Activate the environment:

   ```bash
   conda activate dsa3101_env
   ```

4. **Run the Python Script**

   With the environment activated, you can now run your script. For example:

   ```bash
   python main.py
   ```

   This command will execute `main.py` in the current environment. Make sure that `main.py` is within the container’s working directory.



## API documentation
### Data:
* From `group_tasks/Subgroup A/qn3(optional).ipynb`:
    * for customer churn likelihood: `data/predictions/A_BQ3.csv`
    * for risk of churning: `data/predictions/A_BQ3_pt2.csv`
* From `group_tasks/Subgroup B/qn1.ipynb`:
    * for recommendation system probabilities: `data/predictions/BQ1.csv`
  
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


### 3) Data Dictionary for BankChurners_more Dataset
| Variable               | Data Type   | Description                                             | Example Value |
|------------------------|-------------|---------------------------------------------------------|---------------|
| CLIENTNUM              | Integer     | Unique identifier for each customer                     | 768805383     |
| Attrition_Flag         | Integer     | Indicates if the customer has churned (1 = Churned, 0 = Existing Customer) | 1             |
| Customer_Age           | Integer     | Age of the customer                                     | 45            |
| Gender                 | Integer     | Gender of the customer (1 = Male, 0 = Female)           | 1             |
| Dependent_count        | Integer     | Number of dependents associated with the customer       | 3             |
| Education_Level        | Categorical | Education level of the customer                         | High School   |
| Marital_Status         | Integer     | Marital status of the customer (1 = Married, 0 = Single) | 1           |
| Income_Category        | Integer     | Income bracket of the customer (e.g., 1 = <$40K, 2 = $40K-60K) | 2 |
| Card_Category          | Integer     | Type/category of card held by the customer (1 = Basic, 2 = Silver) | 1 |
| Month_with_bank        | Integer     | Number of months the customer has been with the bank    | 39            |
| No_of_product          | Integer     | Number of products the customer is using                | 5             |
| Months_Inactive_12_mon | Integer     | Number of months the customer was inactive in the past 12 months | 1   |
| Credit_Limit           | Float       | Credit limit for the customer                           | 12691.0       |
| Total_Revolving_Bal    | Float       | Total revolving balance for the customer                | 777           |
| Total_Trans_Amt        | Float       | Total transaction amount                                | 1144          |
| Total_Trans_Count      | Integer     | Total count of transactions                             | 42            |


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
