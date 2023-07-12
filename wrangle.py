
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from pydataset import data
 

#import ignore warnings
import warnings
warnings.filterwarnings("ignore")



##############################################   ACQUIRE     ##############################################


# Read data from the zillow table in the zillow database on our mySQL server.
from env import host, user, password

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
    
##############################################    CLEAN     ######################################################



def wrangle_zillow():
    '''
    Read zillow data into a pandas DataFrame from mySQL,
    drop columns not needed for analysis, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned zillow DataFrame.
    '''

    zillow = get_zillow_data()

    # Replace white space values with NaN values.
    zillow = zillow.replace(r'^\s*$', np.nan, regex=True)

    # Drop any rows with NaN values.
    df = zillow.dropna()
    
      #rename the columns
    df = df.rename(columns={'bedroomcnt':'bedcount',
                        'bathroomcnt':'bathcount',
                        'calculatedfinishedsquarefeet': 'sqft',
                        'taxvaluedollarcnt': 'tax_value',})

    # change the dtype for the necessary columns
    df['sqft'] = df.sqft.astype(int)

    df['yearbuilt'] = df.yearbuilt.astype(int)

    df['sqft'] = df.sqft.astype(int)

    df['fips'] = df.fips.astype(int).astype(str)

    
   
    return df
 ##############################################   TRAIN TEST VALIDATE   ##############################################


def min_max_scaler(train, validate, test):
    '''
    Scale the features in train, validate, and test using MinMaxScaler.

    Args:
        train (DataFrame): The training data.
        validate (DataFrame): The validation data.
        test (DataFrame): The test data.

    Returns:
        scaler (object): The MinMaxScaler object.
        train_scaled (DataFrame): The scaled training data.
        validate_scaled (DataFrame): The scaled validation data.
        test_scaled (DataFrame): The scaled test data.
    '''

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train))
    validate_scaled = pd.DataFrame(scaler.transform(validate))
    test_scaled = pd.DataFrame(scaler.transform(test))

    return train_scaled, validate_scaled, test_scaled

##############################################   SPLIT   ############################################################


def split_zillow(df):
    '''This function splits the clean zillow data stratified on '''
    
    
  # train/validate/test split
    
   
    train_validate, test = train_test_split(df, test_size = .2, random_state=311)

    train, validate = train_test_split(train_validate, test_size = .25, random_state=311)

    return train, validate, test


