import pandas as pd
import ML_ModelBuild_Predict_Evaluate_v1 as mlb

test_csv_file = 'C:/Users/i308124/Desktop/babu project/new_crop_data.csv'
output_prediction_file = 'C:/Users/i308124/Desktop/babu project/prediction.csv'


def process_predict_new_data():
    new_data = pd.read_csv(test_csv_file)
    print("*****New data for which crop yield prediction needed*****\n")
    print(new_data.head())

    new_data_processed = new_data.iloc[:, 1:20]

    new_data_processed["District"] = new_data_processed["District"].\
        replace(["Ahmedabad", "Banglore", "Banswara", "Bhagalpur", "Coimbatore", "Dharwad", "Gulberga", "Hisar", "Indore","Jaipur", "Jalna", "Rewari", "Tonk"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    new_data_processed["Season"] = new_data_processed["Season"].\
        replace(["Kharif", "Rabi", "Summer", "Whole year"], [1, 2, 3, 4])
    new_data_processed["Crop"] = new_data_processed["Crop"].\
        replace(["Sunflower", "Groundnut", "Rapeseed & mustard", "Wheat", "Guar seed", "Bajra", "Jowar", "Turmeric", "Arhar/tur", "Horse-gram", "Other  rabi pulses", "Mesta", "Sugarcane", "Gram", "Tobacco", "Coriander", "Moth", "Garlic", "Banana", "Dry chillies", "Onion", "Other kharif pulses", "Soyabean", "Rice", "Urad", "Niger seed", "Potato", "Castor seed", "Moong", "Maize", "Small millets", "Sannhamp", "Cotton", "Safflower", "Masoor", "Sweet potato", "Sesamum", "Arecanut", "Linseed", "Tapioca", "Ragi", "Barley", "Black pepper", "Cashewnut", "Mustard", "Peas & beans (pulses)", "Cardamom", "Ginger", "Khesari", "Other rabi pulses", "Sanhump", "Other oilseeds", "Peas", "Jute", "Cowpea"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55])

    print("*****Data cleansing result for new data*****\n")
    print(new_data_processed.head())

    new_data_processed.fillna(new_data_processed.mean(), inplace=True)

    print("crop yield prediction using Decision Tree model - new data\n")
    new_data['yield prediction from DT'] = mlb.DT_regressor.predict(new_data_processed)
    mlb.DT_regressor.predict(new_data_processed)
    print(new_data.head(15))

    print("crop yield prediction using Multiple Linear Regression Model - new data \n")
    new_data['yield prediction from MLR'] = mlb.linear_regressor.predict(new_data_processed)
    print(new_data.head(15))

    #
    # save result in prediction file
    output_prediction_data = new_data.iloc[:, [0, 20, 21]]
    print(output_prediction_data.head(15))
    output_prediction_data.to_csv(output_prediction_file, index=False, encoding='utf-8')


