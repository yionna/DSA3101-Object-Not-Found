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

### Environment setup

### Running the code

## Repository structure
```
personalized-marketing-bank/
├──                         # Modular Python scripts for data cleaning, analysis, modelling, and visualization
├── main.py                 # Orchestrates the entire data pipeline and analysis process
├── config.py               # For all configuration parameters
├── utils.py                # For utility functions used across multiple scripts
├──                         # SQL scripts for data extraction and transformation
├──                         # Simple API (using Flask or FastAPI) to serve model predictions and key insights
├── requirements.txt        # All dependencies
├──                         # For all functions, classes, and modules
├──                         # Dockerfile to containerize the application
├── data/                   # Contains raw and preprocessed data files
│   ├── 
│   ├── 
│   ├── 
│   └── 
├── src/                    # Source code for the application
│   └──
└── README.md               # Project documentation
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

### Request/Response fomrats
