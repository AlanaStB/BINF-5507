#“I used ChatGPT  to help troubleshoot errors for this assignment.”

# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    data = data.loc[:, data.isnull().mean() < 0.25].copy()  # Remove columns with more than 75% missing values
    #seperate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    # filling missing numerical values by method selected
    for col in numerical_cols:
         if strategy == 'mean':
             data[col] = data[col].fillna(data[col].mean()) 
         elif strategy == 'median':
            data[col] = data[col].fillna(data[col].median())
         elif strategy == 'mode':
            data[col] = data[col].fillna(data[col].mode()[0])
    # filling missing categorical values by mode
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    return data  # Return the DataFrame with imputed values
# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    data = data.drop_duplicates()
    return data  # Remove duplicate rows


# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    elif method == 'standard':
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data  


# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    #Create a correlation matrix
    categorical_cols = data.select_dtypes(include=['object']).columns
    # Convert categorical variables so they can be processed by the corr function.
    cor_data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    corr_matrix = cor_data.corr()
    # the correlation with measure the same column against itself, so we need to remove the duplicates
    corr_nodups = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_features = []
    # Identify columns with correlation above the threshold
    for r in range(corr_nodups.shape[0]):
        for c in range(corr_nodups.shape[1]):
            if abs(corr_nodups.iloc[r, c]) >threshold:
                colname = corr_nodups.columns[r]
                redundant_features.append(colname)
    print(f'Redundant features identified: {redundant_features}')
    # Drop the redundant columns
    data = data.drop(columns=redundant_features)
    
    return data  

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None