import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def model_train_test(df, cols, product_list, clusters):
    training_features = {key: {} for key in clusters.keys()}
    predicted_labels = {key: {} for key in clusters.keys()}
    actual_labels = {key: {} for key in clusters.keys()}

    # Loop through clusters and products
    for key, cluster in clusters.items():
        users = df[df['Segment'].isin(cluster)]
        for i in product_list:
            # Define the target variable for the product
            y = users[i]
            
            # Set up training features from selected columns
            training_features[key][i] = users[list(cols[key][i]) + ['CLIENTNUM']]
            training_features[key][i] = training_features[key][i].set_index('CLIENTNUM')
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(training_features[key][i], y, test_size=0.4, random_state=123)
            
            # Train XGBClassifier
            bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
            bst.fit(X_train.to_numpy(), y_train.to_numpy())
            
            # Prediction
            predicted_labels[key][i] = pd.DataFrame(bst.predict_proba(X_test.to_numpy())[:, 1], index=X_test.index, columns=[i])
            actual_labels[key][i] = pd.DataFrame(y_test, index=X_test.index, columns=['Actual'])
    return training_features, predicted_labels, actual_labels