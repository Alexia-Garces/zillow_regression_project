import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

###################### Acquire Zillow Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = """
    SELECT p.parcelid AS parcel_id,
	    taxvaluedollarcnt AS tax_value,
        bathroomcnt AS bathroom_cnt,
        bedroomcnt AS bedroom_cnt,
        calculatedfinishedsquarefeet AS sqft_calculated,
        poolcnt AS has_pool,
        p.fips AS fips,
        taxamount AS tax_amount,
        transactiondate AS transaction_date,
        yearbuilt AS year_built
    FROM properties_2017 AS p
    JOIN predictions_2017 AS pred ON p.`parcelid` = pred.`parcelid`
	WHERE p.`propertylandusetypeid` IN (261) 
    	AND pred.`transactiondate` BETWEEN '2017-05-01' AND '2017-08-31'
        AND bathroomcnt > 0
        AND calculatedfinishedsquarefeet > 0 
        AND taxamount > 0
        AND taxvaluedollarcnt > 0
        AND fips > 0
    ORDER BY fips;
    """

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    return df

###################### Prepare Zillow Data ######################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    
    return train, validate, test

def prep_zillow_taxrate(df):
    
    """
    This function takes in the zillow dataframe and retuns the cleaned and prepped dataset
    to use when doing exploratory data analysis
    """
    
    #convert column data types
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    df.has_pool = df.has_pool.astype(object)
    df.parcel_id = df.parcel_id.astype(object)
    
    #change transaction date to datetime
    df['transaction_date'] = pd.to_datetime(df.transaction_date)
    
    #create a tax_rate column
    df["tax_rate"] = df["tax_amount"] / df["tax_value"]*100
    
    #remove outliers
    df = remove_outliers(df, 1.7, ['tax_value', 'bathroom_cnt', 'sqft_calculated', 'tax_rate'])

    #create a county name column on df
    df["county_name"] = df["fips"].map({6037: "Los Angeles", 6059: "Orange", 6111: "Ventura"})

    #create csv for tax_rate
    df.to_csv('zillow_tax.csv')

    return df

def prepare_zillow(df):
    
    """
    This function takes in the zillow dataframe and retuns the cleaned and prepped dataset
    to use when doing exploratory data analysis
    """
    
    #remove outliers
    df = remove_outliers(df, 1.7, ['bedroom_cnt', 'bathroom_cnt', 'sqft_calculated', 'tax_value'])
    
    #fill in nulls for pool with zero
    df['has_pool'] = df.has_pool.fillna(value=0)

    #convert column data types
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    df.has_pool = df.has_pool.astype(object)
    df.parcel_id = df.parcel_id.astype(object)
    
    #change transaction date to datetime
    df['transaction_date'] = pd.to_datetime(df.transaction_date)

    #drop taxamount column
    df = df.drop(columns=['tax_amount'])
    
    #split data
    train, validate, test = split_continuous(df)
    
    #impute year_built with mode
    #create imputer
    imputer = SimpleImputer(strategy='most_frequent')
    #fit to train
    imputer.fit(train[['year_built']])
    #transform data
    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])
    
    return train, validate, test

def wrangle_zillow():
    """
    This functions acquires the zillow data and retuns the cleaned and prepped dataframe
    to use when doing exploratory data analysis
    """
    train, validate, test = prepare_zillow(get_zillow_data())
    return train, validate, test

############################### Scaler Function##################################
def tip_the_scale(train, validate, test, column_names, scaler, scaler_name):
    
    '''
    This function takes in the train validate and test dataframes, list of columns you want to scale, a scaler type,
    scaler_name
    column_names: list of columns to scale
    scaler_name, the name for the new dataframe columns
    adds columns to the train validate and test dataframes
    outputs scaler for doing inverse transforms
    ouputs a list of the new column names
    
    '''
    
    #create the scaler (input here should be scaler type used)
    mm_scaler = scaler
    
    #make empty list for return
    scaled_column_list = []
    
    #loop through columns in col names
    for col in column_names:
        
        #fit and transform to train, add to new column on train df
        train[f'{col}_{scaler_name}'] = mm_scaler.fit_transform(train[[col]]) 
        
        #df['col'].values.reshape(-1, 1)
        
        #transform cols from validate and test (only fit on train)
        validate[f'{col}_{scaler_name}']= mm_scaler.transform(validate[[col]])
        test[f'{col}_{scaler_name}']= mm_scaler.transform(test[[col]])
        
        #add new column name to the list that will get returned
        scaled_column_list.append(f'{col}_{scaler_name}')
    
    #returns scaler, and a list of column names that can be used in X_train, X_validate and X_test.
    return scaler, scaled_column_list 