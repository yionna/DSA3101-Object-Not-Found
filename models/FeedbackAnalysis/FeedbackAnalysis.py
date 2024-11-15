import pandas as pd
from textblob import TextBlob
from collections import Counter

def comment_analysis(date, title, comment):
    '''
    This function saves the basic information of a comment which includes its published date, title and content.
    Meanwhile, it also saves the sentiment polarity of the comment and the themes extracted from it.

    Args:
        date (string): The published date of a comment
        title (string): The title of a comment
        comment (string): The content of a comment
    
    Returns:
        dict: a dictionary contains the published date, the title, the content, the sentiment polarity and the themes extracted of a comment
    '''
    dict = {}
    dict['date'] = date
    dict['title'] = title
    dict['comment'] = comment

    analysed_title = TextBlob(title)
    analysed_comment = TextBlob(comment)

    # sentiment analysis
    dict['polarity'] = (analysed_title.sentiment.polarity + analysed_comment.sentiment.polarity)/2
    
    # theme extraction
    dict['themes'] = analysed_title.noun_phrases + analysed_comment.noun_phrases

    return dict

def construct_dataset(dates, titles, comments):
    '''
    This function constructs a DataFrame which contains the basic information and the result of NLP analysis of comments collected.

    Args:
        dates (datetime): The list of published dates of comments collected
        titles (string): The list of title of comments collected
        comments (string): The list of the content of comments collected

    Returns:
        DataFrame: A table of five columns. 
                   Each row represents a column and contains its published date, title, content, sentiment polarity and the themes extracted from it.
    '''
    dataset = []

    for date, title, comment in zip(dates, titles, comments):
        dataset.append(comment_analysis(date, title, comment))
    
    return pd.DataFrame(dataset)

def sentiment_categorisation(nlp_dataset):
    '''
    This function categorises comments based on the sentiment score.

    Args:
        nlp_dataset (DataFrame): A dataset of comments that is construsted based on comment_analysis() and construct_dataset().
                                 It should have a `polarity` column that contains sentiment scores of comments

    Returns:
        Series: A new column that contains the sentiment category of the comments.
                It can be added to a DataFrame.
    '''
    negative_polarity = nlp_dataset[nlp_dataset['polarity'] < 0]['polarity']
    positive_polarity = nlp_dataset[nlp_dataset['polarity'] > 0]['polarity']

    q_neg = negative_polarity.quantile(0.10)
    q_neu_lower = negative_polarity.quantile(0.90)
    q_neu_upper = positive_polarity.quantile(0.10)
    q_pos = positive_polarity.quantile(0.90)

    bins = [-1, q_neg, q_neu_lower, q_neu_upper, q_pos, 1]
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

    polarity_category = pd.cut(nlp_dataset['polarity'], bins=bins, labels=labels)

    return polarity_category

def theme_selection(nlp_dataset, min_count):
    '''
    This function extracts themes that are more prevalent from a dataset of comments.

    Args:
        nlp_dataset (DataFrame): A dataset of comments that is construsted based on based on comment_analysis() and construct_dataset().
                                 It should have a `theme` column that contains themes extracted from the comments.
        min_count: The benchmark for the themes to be deemed as prevalent.

    Returns:
        DataFrame: A table of two columns, `Theme` and `Count` (the count of the theme in the same row).
    '''

    theme_list = [theme for list in nlp_dataset['themes'] for theme in list]
    theme_counts = Counter(theme_list)

    # only keep the themes which have more count than the min_count
    filtered_theme_counts = {theme: count for theme, count in theme_counts.items() if count >= min_count}

    theme_df = pd.DataFrame(filtered_theme_counts.items(), columns=['Theme', 'Count'])

    return theme_df