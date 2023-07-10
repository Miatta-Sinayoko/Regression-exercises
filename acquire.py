#Import Libraries
import pandas as pd
import numpy as np
import os
from pydataset import data
from sqlalchemy import create_engine

# Acquire
from env import host, user, password

# Create a function that retrieves the necessary connection URL.

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# Create function to retrieve zillow data

def get_zillow_data():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    
    filename = 'zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        
        sql = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
                FROM  properties_2017
                WHERE propertylandusetypeid = 261
                '''
        df.to_csv(filename, index=False)

        df = pd.read_sql(sql, get_connection('zillow'))
        
        return df

