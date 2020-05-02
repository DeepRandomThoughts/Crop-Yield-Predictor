import pandas as pd
import ML_ModelBuild_Predict_Evaluate as mlb
import numpy as np

def process_predict_newdata():
    new_data = pd.read_csv('C:/Users/i308124/Desktop/babu project/new_crop_data.csv')
    print("*****New data for which crop yield prediction needed*****\n")
    print(new_data.head())

    new_data_processed = new_data.iloc[:, 1:9]
    # new_data_processed["Location"] = new_data_processed["District"].replace(["Mysore", "Mandya", "Raichur", "Koppal"],
    #                                                                         [1, 2, 3, 4])

    new_data_processed=new_data
    new_data_processed["District"] = new_data_processed["District"].replace(["Ahmedabad", "Banglore", "Banswara", "Bhagalpur", "Coimbatore", "Dharwad", "Gulberga", "Hisar", "Indore","Jaipur", "Jalna", "Rewari", "Tonk"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    new_data_processed["Season"] = new_data_processed["Season"].replace(["Kharif", "Rabi", "Summer", "Whole year"], [1, 2, 3, 4])
    new_data_processed["Crop"] = new_data_processed["Crop"].replace(["Sunflower","Groundnut","Rapeseed & mustard","Wheat","Guar seed","Bajra","Jowar","Turmeric","Arhar/tur","Horse-gram","Other  rabi pulses","Mesta","Sugarcane","Gram","Tobacco","Coriander","Moth","Garlic","Banana","Dry chillies","Onion","Other kharif pulses","Soyabean","Rice","Urad","Niger seed","Potato","Castor seed","Moong","Maize","Small millets","Sannhamp","Cotton","Safflower","Masoor","Sweet potato","Sesamum","Arecanut","Linseed","Tapioca","Ragi","Barley","Black pepper" ,"Cashewnut" ,"Mustard" ,"Peas & beans (pulses)" ,"Cardamom" ,"Ginger" ,"Khesari" ,"Other rabi pulses" ,"Sanhump" ,"Other oilseeds" ,"Peas" ,"Jute", "Cowpea"], [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55])

    # new_data_processed = new_data_processed.fillna(new_data_processed.mean())

    # new_data_processed=pd.DataFrame(new_data_processed).fillna(0)
    # new_data_processed = (new_data_processed - new_data_processed.mean()) / (
    #             new_data_processed.max() - new_data_processed.min())

    print("*****Data cleanising result for new data*****\n")
    print(new_data_processed.head())

    print("-----------dev debug----------")
    # print(np.where(np.isnan(new_data_processed)))
    # new_data_processed = new_data_processed.fillna(new_data_processed.mean())
    new_data_processed.fillna(new_data_processed.mean(), inplace=True)

    # ohe = OneHotEncoder()
    # ohe.fit(new_data_processed)
    # new_data_processed = ohe.transform(new_data_processed)

    # np.where(new_data_processed.values >= np.finfo(np.float64).max)

    # new_data_processed=clean_dataset(new_data_processed)
    new_data_processed.to_csv("C:/Users/i308124/Desktop/babu project/processed_data_test.csv", index=False)
    # new_data_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("crop yield prediction using Decision Tree model - new data\n")

    new_data['yield prediction from DT'] = mlb.DT_regressor.predict(new_data_processed)
    mlb.DT_regressor.predict(new_data_processed)
    print(new_data.head(15))

    # print("crop yield prediction using Multiple Linear Regression Model - new data \n")
    # new_data['yield prediction from MLR'] = mlb.linear_regressor.predict(new_data_processed)
    # print(new_data.head(15))
    #
    # # save result in file
    # filename = 'C:/Users/i308124/Desktop/babu project/Crop_yield_predictionfornewdata.csv'
    # new_data.to_csv(filename, index=False, encoding='utf-8')


