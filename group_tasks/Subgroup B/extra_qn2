import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sklearn.model_selection import train_test_split
# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Sample customer data
# customer_data = pd.DataFrame({
#     'CustomerID': [1, 2, 3],
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [28, 34, 45],
#     'Gender': ['Female', 'Male', 'Male'],
#     'Interests': ['Fitness, Health', 'Technology, Gadgets', 'Travel, Photography']
# })

# Combine relevant features into a personalized prompt
def create_prompt(row):
    prompt = f"Create a marketing message for a {row['Age']} year old {row['Gender']} interested in {row['Interests']}."
    return prompt

customer_data['Prompt'] = customer_data.apply(create_prompt, axis=1)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate messages
def generate_message(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return message

customer_data['MarketingMessage'] = customer_data['Prompt'].apply(generate_message)

# Evaluate messages
sentiment_analyzer = SentimentIntensityAnalyzer()

def evaluate_message(message):
    sentiment = sentiment_analyzer.polarity_scores(message)
    sentiment_score = sentiment['compound']
    readability = flesch_reading_ease(message)
    sentences = sent_tokenize(message)
    num_sentences = len(sentences)
    return pd.Series({
        'SentimentScore': sentiment_score,
        'ReadabilityScore': readability,
        'NumSentences': num_sentences
    })

customer_data[['SentimentScore', 'ReadabilityScore', 'NumSentences']] = customer_data['MarketingMessage'].apply(evaluate_message)

# Save the results
customer_data.to_csv('personalized_marketing_messages_evaluation.csv', index=False)

# Display a sample of the results
print(customer_data[['CustomerID', 'MarketingMessage', 'SentimentScore', 'ReadabilityScore']])
