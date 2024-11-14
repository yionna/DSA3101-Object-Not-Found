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

-- This is a join table between banking_behavior and bank_churners
-- In reality, this demographic shall be separated as demographic(reference to bank_churners) and banking_behaviors
-- Table: Customer_Demographics
CREATE TABLE IF NOT EXISTS Customer_Demographics (
    Churn TINYINT,                                -- 1 = Existing, 0 = AttritedCLIENTNUM INT PRIMARY KEY,                            -- Unique identifier for each customer
    Customer_Age INT NOT NULL,                            -- Age of the customer
    Gender TINYINT CHECK (Gender IN (0, 1)),              -- 0 = Female, 1 = Male
    Dependent_count INT,                                  -- Number of dependents
    Education_Level TINYINT CHECK (Education_Level BETWEEN 1 AND 5), -- Example: 1 = High School, 2 = Graduate, etc.
    Marital_Status TINYINT CHECK (Marital_Status BETWEEN 1 AND 3),   -- Example: 1 = Married, 2 = Single, etc.
    Income_Category TINYINT CHECK (Income_Category BETWEEN 1 AND 5), -- Income bracket as categories
    Total_Trans_Count INT,                                 -- Total count of transactions
    Total_Trans_Amt FLOAT,                                 -- Total transaction amount for the customer
    Avg_Utilization_Ratio FLOAT,                           -- Average credit utilization ratio
    Savings FLOAT,                                         -- Savings balance of the customer
    Balance FLOAT,                                         -- Current account balance
    Credit_Score INT,                                      -- Credit score of the customer
    Outstanding_Loans FLOAT,                               -- Total outstanding loans
    Digital_Capability BOOLEAN,                            -- Indicates if the customer uses digital services (1 = Yes, 0 = No)
    Loyalty VARCHAR(50),                                   -- Loyalty category
    Financial_Status VARCHAR(50),                          -- Financial status of the customer
    Segment VARCHAR(50)                                  -- Segment type for the customer
);


-- Table: Transaction_Data
CREATE TABLE IF NOT EXISTS Transaction_Data (
    Transaction_ID INTEGER PRIMARY KEY AUTO_INCREMENT,    -- Unique identifier for each transaction
    CLIENTNUM INTEGER NOT NULL,                           -- Foreign key to identify customer
    Transaction_Date DATETIME NOT NULL,                   -- Date of the transaction
    Transaction_Time TIMESTAMP NOT NULL,                  -- Time of the transaction
    Transaction_Amount FLOAT NOT NULL,                    -- Amount of the transaction
    CustAccountBalance FLOAT,                             -- Customer's account balance after the transaction
    FOREIGN KEY (CLIENTNUM) REFERENCES Customer_Demographics(CLIENTNUM)  -- Links to the customer demographics table
);

-- Table: Campaign_Database
CREATE TABLE IF NOT EXISTS Campaign_Database (
    CLIENTNUM INTEGER NOT NULL,                           -- Foreign key to identify customer
    Campaign_ID INTEGER NOT NULL,                         -- Unique identifier for each campaign
    Campaign_Type VARCHAR(100),                           -- Type of campaign
    Product VARCHAR(100),                                 -- Product associated with the campaign
    Start_Date DATETIME,                                  -- Start date of the campaign
    End_Date DATETIME,                                    -- End date of the campaign
    ResponseStatus VARCHAR(50),                           -- Response status of the customer
    NumberOfImpressions INT,                              -- Number of impressions the campaign received
    NumberOfClicks INT,                                   -- Number of clicks the campaign received
    PRIMARY KEY (CLIENTNUM, Campaign_ID),                 -- Composite primary key
    FOREIGN KEY (CLIENTNUM) REFERENCES Customer_Demographics(CLIENTNUM),  -- Links to the customer demographics table
    FOREIGN KEY (Campaign_ID) REFERENCES Campaign_log(Campaign_ID)        -- Links to the campaign log table
);

-- Table: Campaign_log
CREATE TABLE IF NOT EXISTS Campaign_log (
    Campaign_ID INTEGER PRIMARY KEY,                      -- Unique identifier for each campaign
    Campaign_Type VARCHAR(100),                           -- Type of campaign
    ChosenChannel VARCHAR(100),                           -- Preferred channel for the campaign
    ChosenFrequency INT,                                  -- Frequency chosen for the campaign
    ChosenTiming INT,                                     -- Timing chosen for the campaign
    ConversionRate FLOAT,                                 -- Conversion rate of the campaign
    ClickThroughRate FLOAT                                -- Click-through rate of the campaign
);

-- Table: Product_Subscribed
CREATE TABLE IF NOT EXISTS Product_Subscribed (
    CLIENTNUM INTEGER,                                    -- Foreign key to identify customer
    credit_cards BOOLEAN,                                 -- Indicates if customer has subscribed to credit card services
    savings BOOLEAN,                                      -- Indicates if customer has savings account
    investments BOOLEAN,                                  -- Indicates if customer has investments
    personal_insurance BOOLEAN,                           -- Indicates if customer has personal insurance
    commercial_insurance BOOLEAN,                         -- Indicates if customer has commercial insurance
    personal_loans BOOLEAN,                               -- Indicates if customer has personal loans
    commercial_loans BOOLEAN,                             -- Indicates if customer has commercial loans
    PRIMARY KEY (CLIENTNUM),                              -- Primary key is CLIENTNUM
    FOREIGN KEY (CLIENTNUM) REFERENCES Customer_Demographics(CLIENTNUM)  -- Links to the customer demographics table
);

-- Display the tables in the database
SHOW TABLES;
