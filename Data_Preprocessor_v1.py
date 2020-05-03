import pandas as pd

preprocessedFile = 'C:/Users/skmit/PycharmProjects/untitled/data/crop_yield_datasource_preprocessed.csv'


# Check for missing values and handle them
def handle_missing_data(df):
    print("*****Number of Missing Data present in data set*****\n")
    print(df.isnull().sum())
    print("\n")
    print("*****Handling Missing Data present in data set*****\n")
    print("*****Performing Mean imputation*****\n")
    df_handled_missing_data = df.fillna(value=df.mean())
    print(df_handled_missing_data.head())
    return df_handled_missing_data


# Perform data transformation by doing Normalization
def normalize_data(df_transformed_data):
    feature_cols = list(df_transformed_data)
    predict_cols = feature_cols[-1:]

    df_target = df_transformed_data[predict_cols]
    df_target.rename({'Yield': 'target_yield'}, axis=1, inplace=True)
    df_transformed_data = df_transformed_data.iloc[:,1:21]
    print(df_target.head())

    df_transformed_data = df_transformed_data.apply(pd.to_numeric, errors='coerce')
    df_transformed_data.fillna(0, inplace=True)

    # df_transformed_data = (df_transformed_data - df_transformed_data.mean()) / (df_transformed_data.max()
    # - df_transformed_data.min())
    print("\n*****Normalized Data*****\n")
    print(df_transformed_data.head())

    # dataframe_norm = df_transformed_data.join(df_target, on='Yield', how='left', lsuffix='_left')
    # dataframe_norm = df_transformed_data.join(df_target)
    # dataframe_norm = df_transformed_data

    print(df_transformed_data.head())
    return df_transformed_data


# Convert categorical values to continous
def handle_categorical_data(df_handled_categoricaldata):
    df_handled_categoricaldata["District"] = df_handled_categoricaldata["District"]\
        .replace(["Ahmedabad","Banglore","Banswara","Bhagalpur","Coimbatore","Dharwad","Gulberga","Hisar","Indore","Jaipur","Jalna","Rewari","Tonk"], [1,2,3,4,5,6,7,8,9,10,11,12,13])
    df_handled_categoricaldata["Season"] = df_handled_categoricaldata["Season"].replace(["Kharif", "Rabi", "Summer", "Whole year"], [1,2,3,4])
    df_handled_categoricaldata["Crop"] = df_handled_categoricaldata["Crop"].replace(["Sunflower", "Groundnut", "Rapeseed & mustard", "Wheat", "Guar seed", "Bajra", "Jowar", "Turmeric","Arhar/tur", "Horse-gram", "Other  rabi pulses", "Mesta", "Sugarcane", "Gram", "Tobacco", "Coriander", "Moth","Garlic", "Banana", "Dry chillies", "Onion", "Other kharif pulses", "Soyabean", "Rice", "Urad", "Niger seed","Potato", "Castor seed", "Moong", "Maize", "Small millets", "Sannhamp", "Cotton", "Safflower", "Masoor","Sweet potato", "Sesamum", "Arecanut", "Linseed", "Tapioca", "Ragi", "Barley", "Black pepper", "Cashewnut","Mustard", "Peas & beans (pulses)", "Cardamom", "Ginger", "Khesari", "Other rabi pulses", "Sanhump","Other oilseeds", "Peas", "Jute", "Cowpea"],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55])

    print("\n*****Handling Categorical Data*****\n")
    print(df_handled_categoricaldata.head())
    return df_handled_categoricaldata


def pre_process_data(df):
    df_handled_missing_data = handle_missing_data(df)

    df_handled_categorical_data = handle_categorical_data(df_handled_missing_data)

    df_transformed_data = normalize_data(df_handled_categorical_data)

    return df_transformed_data
