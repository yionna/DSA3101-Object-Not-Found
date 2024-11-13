import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def select_best(x, y, k = 10, score_func=f_regression):
    '''
    Select k best feature with SelectKBest
    Input:
    x: training features
    y: target variable
    Optional:
    k = number of training features selected
    score_func: score function used to evaluate features
    Returns:
    list of k best features, dataframe of scores and p-value of all features
    '''
    KBest = SelectKBest(score_func, k=k)
    KBest = KBest.fit(x, y)

    cols = KBest.get_support(indices=True)
    cols = x.columns[cols]

    return np.array(cols), pd.DataFrame({'features': x.columns, 'score': KBest.scores_, 'p-value': KBest.pvalues_ })


def obtain_best_columns(df, clusters, product_list):
    '''
    Obtain best columns for each product in each cluster
    Input:
    df: dataframe
    clusters: dictionary of clusters
    product_list: list of products
    Returns:
    dictionary of list of best features, dictionary of dataframe of scores and p-value of all features
    '''
    scores = {key:{} for key in clusters.keys()}
    cols = {key:{} for key in clusters.keys()}
    for key, cluster in clusters.items():
        df_cluster = df[df['Segment'].isin(cluster)].drop(columns=['CLIENTNUM'])

        for product in product_list:
            X = df_cluster.drop(columns=[product])
            cols[key][product], scores[key][product] = select_best(X, df_cluster[product])
    return cols, scores


def model_train_test(df, cols, clusters, product_list):
    '''
    Train XGBoost models and predict output
    Input:
    df: dataframe
    cols: columns for training
    clusters: dictionary of clusters
    product_list: list of products
    Returns:
    dictionary of predicted labels, dictionary of ground truth labels
    '''
    predicted_labels = {key: {} for key in clusters.keys()}
    actual_labels = {key: {} for key in clusters.keys()}

    # Loop through clusters and products
    for key, cluster in clusters.items():
        users = df[df['Segment'].isin(cluster)]
        for i in product_list:
            # Define the target variable for the product
            y = users[['CLIENTNUM', i]].set_index('CLIENTNUM')
            
            # Set up training features from selected columns
            training_features = users[list(cols[key][i]) + ['CLIENTNUM']]
            training_features = training_features.set_index('CLIENTNUM')
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(training_features, y, test_size=0.4, random_state=123)
            
            # Train XGBClassifier
            bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
            bst.fit(X_train.to_numpy(), y_train.to_numpy())
            
            # Prediction
            predicted_labels[key][i] = pd.DataFrame(bst.predict_proba(X_test.to_numpy())[:, 1], index=X_test.index, columns=[i])
            actual_labels[key][i] = pd.DataFrame(y_test, index=y_test.index, columns=[i])
    return predicted_labels, actual_labels

