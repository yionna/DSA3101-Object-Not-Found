-- setup_bank_db.sql

-- Create the bank database
CREATE DATABASE IF NOT EXISTS bank;
USE bank;

-- Table: bank_churners
CREATE TABLE IF NOT EXISTS bank_churners (
    CLIENTNUM BIGINT PRIMARY KEY,                    -- Unique identifier for each customer
    Attrition_Flag TINYINT,                          -- 1 = Existing, 0 = Attrited
    Customer_Age INT,                                -- Age of the customer
    Gender TINYINT,                                  -- 1 = Male, 0 = Female
    Dependent_count INT,                             -- Number of dependents
    Education_Level VARCHAR(50),                     -- Highest education level
    Marital_Status TINYINT,                          -- 1 = Married, 0 = Single
    Income_Category TINYINT,                         -- Income bracket (e.g., 2 = 60-80, 4 = Less than 40)
    Card_Category TINYINT,                           -- Type of credit card (e.g., 2 = Gold, etc.)
    Month_with_bank INT,                             -- Number of months with the bank
    No_of_product INT,                               -- Number of products customer is using
    Months_Inactive_12_mon INT,                      -- Number of inactive months in the last 12 months
    Credit_Limit DECIMAL(10, 2),                     -- Credit limit assigned
    Total_Revolving_Bal DECIMAL(10, 2),              -- Total revolving balance
    Total_Trans_Amt DECIMAL(15, 2),                  -- Total transaction amount
    Total_Trans_Count INT,                           -- Total count of transactions
    Avg_Utilization_Ratio DECIMAL(5, 3)              -- Average credit utilization ratio
);

-- Table: banking_behaviour_preference
CREATE TABLE IF NOT EXISTS banking_behaviour_preference (
    CLIENTNUM BIGINT PRIMARY KEY,                    -- Unique identifier for each customer
    Income_Category TINYINT,                         -- Income bracket (e.g., 4 = Less than 40)
    No_of_product INT,                               -- Number of products the customer is using
    Total_Trans_Amt DECIMAL(15, 2),                  -- Total transaction amount for the customer
    Total_Trans_Count INT,                           -- Total count of transactions
    Credit_Score DECIMAL(5, 2),                      -- Credit score of the customer
    Outstanding_Loans DECIMAL(15, 2),                -- Total amount of outstanding loans
    Balance DECIMAL(15, 2),                          -- Current account balance
    PhoneService TINYINT,                            -- Phone service (1 = Yes, 0 = No)
    InternetService TINYINT,                         -- Internet service (1 = Yes, 0 = No)
    TechSupport TINYINT,                             -- Tech support (2 = High, 1 = Moderate, 0 = None)
    PaperlessBilling TINYINT,                        -- Paperless billing (1 = Yes, 0 = No)
    PaymentMethod TINYINT                            -- Payment method (encoded as integers, e.g., 2 = Credit Card)
);

-- Table: app_interaction_data
CREATE TABLE IF NOT EXISTS app_interaction_data (
    CLIENTNUM INT,                                     -- Foreign key to identify customer
    Interaction_Timestamp DATETIME,                    -- Timestamp of the interaction
    Page_Viewed VARCHAR(100),                          -- Page viewed by the customer
    Time_Spent FLOAT,                                  -- Time spent on the page
    PRIMARY KEY (CLIENTNUM, Interaction_Timestamp)     -- Composite primary key
);

-- Table: campaign_database
CREATE TABLE IF NOT EXISTS campaign_database (
    CLIENTNUM INT,                                     -- Foreign key to identify customer
    Campaign_ID INT,                                   -- Unique identifier for each campaign
    Campaign_Type VARCHAR(100),                        -- Type of campaign
    Start_Date DATETIME,                               -- Start date of the campaign
    End_Date DATETIME,                                 -- End date of the campaign
    Response_Status VARCHAR(50),                       -- Response status of the customer
    PRIMARY KEY (CLIENTNUM, Campaign_ID)               -- Composite primary key
);

-- Table: campaign_parameters
CREATE TABLE IF NOT EXISTS campaign_parameters (
    Campaign_ID INT PRIMARY KEY,                       -- Primary key, related to campaigns
    Campaign_Type VARCHAR(100),                        -- Type of campaign
    threshold FLOAT,                                   -- Threshold for the campaign
    amount FLOAT,                                      -- Amount associated with the campaign
    credit_limit FLOAT,                                -- Credit limit offered
    interest_rate FLOAT,                               -- Interest rate offered
    loan_term INT,                                     -- Loan term duration
    loyalty_cat VARCHAR(50),                           -- Loyalty category
    price FLOAT,                                       -- Price associated with the campaign
    coverage FLOAT                                     -- Coverage amount or percentage
);

-- Table: customer_demographic
CREATE TABLE IF NOT EXISTS customer_demographic (
    CLIENTNUM INT PRIMARY KEY,                         -- Unique identifier for each customer
    Customer_Age INT,                                  -- Age of the customer
    Gender VARCHAR(10),                                -- Gender of the customer (e.g., Male/Female)
    Dependent_count INT,                               -- Number of dependents
    Education_Level VARCHAR(50),                       -- Highest education level
    Marital_Status VARCHAR(50),                        -- Marital status of the customer
    Digital_Capability TINYINT(1),                     -- Digital capability (e.g., 1 = Yes, 0 = No)
    Income_Category VARCHAR(50),                       -- Income bracket
    Loyalty VARCHAR(50),                               -- Loyalty category
    Financial_Status VARCHAR(50)                       -- Financial status of the customer
);

-- Table: transaction_data
CREATE TABLE IF NOT EXISTS transaction_data (
    CLIENTNUM INT,                                     -- Foreign key to identify customer
    Transaction_Date DATETIME,                         -- Date of the transaction
    Transaction_Time TIMESTAMP,                        -- Time of the transaction
    Transaction_Amount FLOAT,                          -- Amount of the transaction
    Transaction_Type VARCHAR(50),                      -- Type of transaction (e.g., debit, credit)
    CustAccountBalance FLOAT,                          -- Customer's account balance after transaction
    Default_Check TINYINT(1),                          -- Indicates if there's a default (e.g., 1 = Yes, 0 = No)
    PRIMARY KEY (CLIENTNUM, Transaction_Date, Transaction_Time)  -- Composite primary key
);