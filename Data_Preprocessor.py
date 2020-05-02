import pandas as pd

df = pd.read_csv('C:/Users/i308124/Desktop/babu project/crop_yield_datasource.csv')

preprocessedFile = 'C:/Users/i308124/Desktop/babu project/crop_yield_datasource_preprocessed.csv'

# Check for missing values and handle them
def Handle_MissingData():
    print("*****Number of Missing Data present in dataset*****\n")
    print(df.isnull().sum())
    print("\n")
    print("*****Handling Missing Data present in dataset*****\n")
    print("*****Performing Mean imputation*****\n")
    df_handled_missingData = df.fillna(value=df.mean())
    print(df_handled_missingData.head())
    return df_handled_missingData

# Perform data transformation by doing Normalization
def Normalize_Data(df_transformed_data):
    feature_cols = list(df_transformed_data)
    predict_cols = feature_cols[-1:]
    print("------------My predict cols------------")
    print(predict_cols)
    df_target = df_transformed_data[predict_cols]
    df_transformed_data = df_transformed_data.iloc[:,1:21]
    print(df_target.head())
    df_transformed_data = df_transformed_data.apply(pd.to_numeric, errors='coerce')
    df_transformed_data.fillna(0, inplace=True)

    df_transformed_data = (df_transformed_data - df_transformed_data.mean()) / (df_transformed_data.max() - df_transformed_data.min())
    print("\n*****Normalized Data*****\n")
    print(df_transformed_data.head())

    dataframe_norm = df_transformed_data.join(df_target,on = 'Yield', how = 'left', lsuffix = '_left')

    print(dataframe_norm.head())
    dataframe_norm.to_csv(preprocessedFile, index=False)
    return dataframe_norm

# Convert categorical values to continous
def handle_categorical_data(df_handled_categoricalData):
    df_handled_categoricalData["District"] = df_handled_categoricalData["District"].replace(["Ahmedabad","Banglore","Banswara","Bhagalpur","Coimbatore","Dharwad","Gulberga","Hisar","Indore","Jaipur","Jalna","Rewari","Tonk"], [1,2,3,4,5,6,7,8,9,10,11,12,13])
    df_handled_categoricalData["Season"] = df_handled_categoricalData["Season"].replace(["Kharif", "Rabi", "Summer", "Whole year"], [1,2,3,4])
    df_handled_categoricalData["Crop"] = df_handled_categoricalData["Crop"].replace(["Sunflower", "Groundnut", "Rapeseed & mustard", "Wheat", "Guar seed", "Bajra", "Jowar", "Turmeric","Arhar/tur", "Horse-gram", "Other  rabi pulses", "Mesta", "Sugarcane", "Gram", "Tobacco", "Coriander", "Moth","Garlic", "Banana", "Dry chillies", "Onion", "Other kharif pulses", "Soyabean", "Rice", "Urad", "Niger seed","Potato", "Castor seed", "Moong", "Maize", "Small millets", "Sannhamp", "Cotton", "Safflower", "Masoor","Sweet potato", "Sesamum", "Arecanut", "Linseed", "Tapioca", "Ragi", "Barley", "Black pepper", "Cashewnut","Mustard", "Peas & beans (pulses)", "Cardamom", "Ginger", "Khesari", "Other rabi pulses", "Sanhump","Other oilseeds", "Peas", "Jute", "Cowpea"],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55])

    print("\n*****Handling Categorical Data*****\n")
    print(df_handled_categoricalData.head())
    return df_handled_categoricalData

def preprocess_Data(df):
    df_handled_missingData=Handle_MissingData()
    df_handled_categoricalData=handle_categorical_data(df_handled_missingData)
    print("------------------my test code--------")
    print(df_handled_categoricalData)
    df_transformed_data= Normalize_Data(df_handled_categoricalData)

    print(df_transformed_data)
    return df_transformed_data