import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
from nltk import sent_tokenize

###This is just a prototype
###It takes too long to run without gpu, please do not run it
class PersonalizedMarketingSystem:
    def __init__(self, demographic_data, campaign_database):
        self.demographic_data = demographic_data
        self.campaign_database = campaign_database
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.language_styles = ['Formal', 'Friendly', 'Humorous', 'Urgency', 'Inspirational', 
                                'Technical', 'Emotional', 'Direct', 'Luxury', 'Educational']
        
        # Set the pad token to eos_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_prompt(self, row, style):
        prompt = f"Create a {style.lower()} marketing message for a {row['Customer_Age']} year old {row['Gender']}."
        if pd.notnull(row['Education_Level']):
            prompt += f" This individual has an education level of {row['Education_Level']}."
        if pd.notnull(row['Marital_Status']):
            prompt += f" They are {row['Marital_Status']}."
        if pd.notnull(row['Income_Category']):
            prompt += f" Their income category is {row['Income_Category']}."
        if pd.notnull(row['Loyalty']):
            prompt += f" This customer has a loyalty status of {row['Loyalty']}."
        return prompt

    def generate_message(self, prompt, max_length=100):
        # Encode the input with return_tensors to get input IDs and attention mask
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id  # Set the pad token ID to the EOS token ID
        )
        message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return message


    def evaluate_message(self, message):
        sentiment = self.sentiment_analyzer.polarity_scores(message)
        sentiment_score = sentiment['compound']
        readability = flesch_reading_ease(message)
        sentences = sent_tokenize(message)
        num_sentences = len(sentences)
        return {
            'SentimentScore': sentiment_score,
            'ReadabilityScore': readability,
            'NumSentences': num_sentences
        }

    def find_best_message(self, row):
        best_message = None
        best_score = -1
        best_style = None
        
        for style in self.language_styles:
            prompt = self.create_prompt(row, style)
            message = self.generate_message(prompt)
            evaluation = self.evaluate_message(message)

            if evaluation['ReadabilityScore'] > 50 and evaluation['SentimentScore'] > 0.5:
                if evaluation['SentimentScore'] > best_score:
                    best_message = message
                    best_score = evaluation['SentimentScore']
                    best_style = style
        
        return best_message, best_style

    def update_campaign_database(self):
        for _, row in self.demographic_data.iterrows():
            clientnum = row['CLIENTNUM']
            best_message = row['BestMessage']
            preferred_style = row['PreferredStyle']
            
            if pd.notnull(best_message):
                self.campaign_database.loc[self.campaign_database['CLIENTNUM'] == clientnum, 'Message'] = best_message
                self.campaign_database.loc[self.campaign_database['CLIENTNUM'] == clientnum, 'PreferredStyle'] = preferred_style

        print("Campaign database updated with marketing messages and customer preferences.")
