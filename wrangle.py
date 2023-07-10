import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env

from env import host, user, password

# Read data from the zillow table in the zillow database on our mySQL server.

def get_connection(db_name):
    
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:

        # Read the SQL query into a dataframe
        df = pd.read_sql('SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 WHERE propertylandusetypeid = 261',get_connection('zillow')) 

        
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df
    

def wrangle_zillow():
    '''
    Read zillow data into a pandas DataFrame from mySQL,
    drop columns not needed for analysis, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned zillow DataFrame.
    '''

    # Acquire data

    zillow = get_zillow_data()

    # Replace white space values with NaN values.
    zillow = zillow.replace(r'^\s*$', np.nan, regex=True)

    # Drop any rows with NaN values.
    df = zillow.dropna()
    
    # Drop Unnamed: 0 column
    df = df.drop(columns = 'Unnamed: 0')

    return df

def min_max_scaler(X_train, X_validate, X_test):
    """
    Scale the features in X_train, X_validate, and X_test using MinMaxScaler.

    Args:
        X_train (DataFrame): The training data.
        X_validate (DataFrame): The validation data.
        X_test (DataFrame): The test data.

    Returns:
        scaler (object): The MinMaxScaler object.
        X_train_scaled (DataFrame): The scaled training data.
        X_validate_scaled (DataFrame): The scaled validation data.
        X_test_scaled (DataFrame): The scaled test data.
    """

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validate_scaled, X_test_scaled
