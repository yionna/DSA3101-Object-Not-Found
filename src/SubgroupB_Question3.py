import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Read in the main dataset
df = pd.read_csv("BankChurners.csv")
#print(df.columns)
'''
Index(['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
       'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
      dtype='object')
'''
#Read in the digital marketing dataset
df3 = pd.read_csv("digital_marketing_campaign_dataset.csv")
#print(df3.columns)
'''
Index(['CustomerID', 'Age', 'Gender', 'Income', 'CampaignChannel',
       'CampaignType', 'AdSpend', 'ClickThroughRate', 'ConversionRate',
       'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares',
       'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints',
       'AdvertisingPlatform', 'AdvertisingTool', 'Conversion'],
      dtype='object')
'''

#Data for df
#Binning Age
bins = [18, 30, 40, 50, 60, 100]
labels = ['18-30', '30-40', '40-50', '50-60', '60+']
df['Age_Group'] = pd.cut(df['Customer_Age'], bins=bins, labels=labels)
print(df['Age_Group'].head())



#Data cleansing for df3
##Removing null
df3 = df3.dropna()
print(df3.shape) #(8000, 20)

#Binning Age
bins = [18, 30, 40, 50, 60, 100]
labels = ['18-30', '30-40', '40-50', '50-60', '60+']
df3['Age_Group'] = pd.cut(df3['Age'], bins=bins, labels=labels)
print(df3['Age_Group'].head())

print(max(df3['Income']))
#Binning Income
bins = [0, 40000, 60000, 80000, 120000, 150000]
labels = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
df3['Income_Category'] = pd.cut(df3['Income'], bins=bins, labels=labels)
print(df3['Income_Category'].head())

#Standardise gender
df3['Gender'] = df3['Gender'] = df3['Gender'].replace(['Male', 'Female'], ['M', 'F'])
print(df3['Gender'].head())

print(df3[['Age_Group','Income_Category','Gender']].value_counts())

#Type of campaign with ad spend and conversion rate
print(df3[['CampaignChannel','CampaignType']].value_counts())
##Adspend
print("Adspend data")
print(df3.groupby(['CampaignChannel', 'CampaignType'])
      ['AdSpend'].agg(['mean', 'max', 'min', 'std']).reset_index())
print("Conversion rate data")
print(df3.groupby(['CampaignChannel', 'CampaignType'])
      ['ConversionRate'].agg(['mean', 'max', 'min', 'std']).reset_index())
###Conclusion: no much difference between the groups of campaign type
###Apply normal distribution to generate data in the main dataframe
print("Adspend data")
print(df3['AdSpend'].agg(['mean', 'max', 'min', 'std']).reset_index())
print("Conversion rate data")
print(df3['ConversionRate'].agg(['mean', 'max', 'min', 'std']).reset_index())

'''
Adspend data
  index      AdSpend
0  mean  5000.944830
1   max  9997.914781
2   min   100.054813
Conversion rate data
  index  ConversionRate
0  mean        0.104389
1   max        0.199995
2   min        0.010018
'''

#Synthesise the data for numeric variables
adspend_stats = {
    'mean': 5000.944830,
    'max': 9997.914781,
    'min': 100.054813,
    'std': 2838.038153
}

conversion_rate_stats = {
    'mean': 0.104389,
    'max': 0.199995,
    'min': 0.010018,
    'std': 0.054878
}

adspend = np.random.normal(
    loc=adspend_stats['mean'],
    scale=adspend_stats['std'],
    size=len(df)
)

adspend = np.clip(adspend, adspend_stats['min'], adspend_stats['max'])

conversion_rate = np.random.normal(
    loc=conversion_rate_stats['mean'],
    scale=conversion_rate_stats['std'],
    size=len(df)
)

conversion_rate = np.clip(conversion_rate, conversion_rate_stats['min'], conversion_rate_stats['max'])

df['AdSpend'] = adspend
df['ConversionRate'] = conversion_rate

print(df[['AdSpend','ConversionRate']].head(5))
print(df[['AdSpend', 'ConversionRate']].describe())


#Synthesise the data for Categorical variables
##Inspect the data structure in df3
categorical = ['CampaignChannel','CampaignType']
for col in categorical:
    print(df3[col].unique())

'''
['Social Media' 'Email' 'PPC' 'Referral' 'SEO']
['Awareness' 'Retention' 'Conversion' 'Consideration']
'''
#Fill in data in df
Campaign_Channel = ['Social Media' ,'Email' ,'PPC' ,'Referral' ,'SEO']
df['Campaign_Channel'] = np.random.choice(Campaign_Channel, size=len(df))

Campaign_Type = ['Awareness' ,'Retention' ,'Conversion' ,'Consideration']
df['Campaign_Type'] = np.random.choice(Campaign_Type, size=len(df))

#Inspection
#print(df[['Campaign_Channel','Campaign_Type']].head(5))


#Calculating the ROI
'''
Source: https://www.marketingevolution.com/marketing-essentials/marketing-roi

Core formula:
Marketing ROI  = (Sales Growth - Marketing Cost) / Marketing Cost

It’s important to note, however, that this formula makes the assumption that
all sales growth is tied to marketing efforts.

Realistic view:
(Sales Growth - Organic Sales Growth - Marketing Cost) / Marketing Cost = Marketing ROI 

'''
'''

'''
#Assumption: average interest earned on deposit by bank: 4.5%
#Average interest paid to depositors: 1.5%

#Calculating the sales growth:
df['Total_Indiv'] = df['Months_on_book']*df['Avg_Open_To_Buy']*(0.045-0.015)

#Inspect
print(df['Total_Indiv'].head(4))

#Conversion rates
average_result = df.groupby('Card_Category')['Avg_Open_To_Buy'].mean().reset_index()

print(average_result)

average_result.rename(columns={'Avg_Open_To_Buy': 'Avg_Open_To_Buy_Avg'}, inplace=True)
df = df.merge(average_result, on='Card_Category')
#Their connections for customers with different card category are different. They tend to meet people with
#similar deposit habits
df['Conversion_Value'] = df['Avg_Open_To_Buy_Avg'] * df['ConversionRate']  # for example, multiply by 2
print(df['Conversion_Value'].head(4))
df['Total_Customer_Growth'] = df['Total_Indiv']+df['Conversion_Value']

#Inspect
print(df['Total_Customer_Growth'].head(4))

#Find ROI
df['ROI'] = (df['Total_Customer_Growth'] - df['AdSpend'])/df['AdSpend']
print(df['ROI'].head(20))



df.to_csv("Campaign_with_ROI.csv")




