import pandas as pd

input_file = 'C:/Users/i308124/Desktop/babu project/crop_yield_datasource.csv'


def display_data(df):
    print("*****Top 5 rows in a data*****")
    print("\n")
    print(df.head())
    print("\n")
    print("*****Bottom 5 rows in a data*****")
    print("\n")
    print(df.tail())
    print("\n")


def describe_data(df):
    print("*****Statistical description of the data*****\n")
    print(df.describe())
    print("\n")
    print("*****Data set Information*****\n")
    print(df.info())


def load_data():
    df = pd.read_csv(input_file)
    display_data(df)
    describe_data(df)
    return df

