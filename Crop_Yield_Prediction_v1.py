import Load_Data_v1 as ld
import Data_Preprocessor_v1 as dp
import ML_ModelBuild_Predict_Evaluate_v1 as ml
import ML_Predictyield_Newdata_v1 as mp


def main():
    df = ld.load_data()

    df_preprocessed = dp.pre_process_data(df)

    ml.perform_machine_learning_tasks(df_preprocessed)

    mp.process_predict_new_data()


if __name__ == "__main__":
    main()
