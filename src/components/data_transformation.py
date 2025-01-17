import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:

    """
        This Class is used for Data Preprocessing, 
        transformations includes:
        - Gender feature mapping, 
        - dummy variable creation, 
        - column rename,
        - type adjustments
        - feature scaling
        - Oversampling using SMOTE-ENN to handle imbalanced dataset.
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
            Initializes the DataTransformation class with the provided artifacts and configuration.
        """
        try:
            # DatIngestionArtifact instance is needed to fetch the train/test data for transformation.
            self.data_ingestion_artifact = data_ingestion_artifact
            # DataTransformationConfig instance is needed to store the transformed data and the used transformer object.
            self.data_transformation_config = data_transformation_config
            # DataValidationArtifact instance is needed to check the validation status before transformation.
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise MyException(e, sys)

    def _map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)

        return df

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')

        return df

    def _drop_unnecessary_column(self, df):
        """Drops the column defined in schema config """
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def feature_scaling_pipeline(self) -> Pipeline:
        """
        Creates and returns a Preprocessing Pipeline object,
        that applies feature scaling to features defined in schema configurations.

        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Load required features from schema configurations
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info(
                "features that need transformation loaded from schema.")

            # Initialize Scalers
            std_scaler = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info(
                "Scalers Initialized: StandardScaler, MinMaxScaler")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", std_scaler, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class")

            return final_pipeline

        except Exception as e:
            logging.exception(
                "Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        Returns:
            DataTransformationArtifact object containing the paths of the transformed data and the transformer object.
        """
        try:
            logging.info("Data Transformation Started !!!")

            # Checks the Data Validation Status
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(
                file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(
                file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Define input and target features for train data
            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # Define input and target features for test data
            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(
                "Input and Target cols separated for both train and test df.")

            # Apply custom transformations in specified sequence to train and test data
            input_feature_train_df = self._map_gender_column(
                input_feature_train_df)
            input_feature_train_df = self._drop_unnecessary_column(
                input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(
                input_feature_train_df)
            input_feature_train_df = self._rename_columns(
                input_feature_train_df)

            input_feature_test_df = self._map_gender_column(
                input_feature_test_df)
            input_feature_test_df = self._drop_unnecessary_column(
                input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(
                input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)

            logging.info(
                "Custom transformations applied to train and test data")

            logging.info("Starting Feature Scaling")
            preprocessor = self.feature_scaling_pipeline()
            logging.info("Got the Scaling pipeline object")

            logging.info(
                "fitting the scaling pipeline on, and applying to, the train data")
            input_feature_train_arr = preprocessor.fit_transform(
                input_feature_train_df)
            logging.info("applying the scaling pipeline to test data")
            input_feature_test_arr = preprocessor.transform(
                input_feature_test_df)

            logging.info(
                "Transformation done end-to-end to train and test df.")

            logging.info(
                "Now Applying SMOTE-ENN for handling imbalanced dataset.")

            # create an instance of SMOTE-ENN
            smt = SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTE-ENN applied to train and test df.")

            # Concatenating input features with target.
            # numpy.c_ is a convenient shorthand used to concatenate arrays column-wise.
            train_arr = np.c_[input_feature_train_final,
                              np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final,
                             np.array(target_feature_test_final)]
            logging.info(
                "feature-target concatenation done for train and test df.")

            save_object(
                self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info(
                "Saved preprocessing object (the scaling pipeline) and preprocessed files.")

            logging.info("Data transformation completed successfully")

            # store the artifact files path.
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
