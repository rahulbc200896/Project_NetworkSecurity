import os
import numpy as np

"Defining variables required for Training Pipeline Configuration"
TARGET_COLUMN = "Result"
PIPELINE_NAME: str = "Network_Security"
ARTIFACT_DIR: str = "Artifacts"
FILENAME: str = "phisingData.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")
MODEL_FILE_NAME = "model.pkl"
PREPROCESSOR_OBJ_FILE = "preprocessor.pkl"


"Defining variables required for Data Ingestion Configuration"
#DATA_INGESTION_DATABASE_NAME :str = ""
#DATA_INGESTION_COLLECTION_NAME :str = ""
DATA_INGESTION_DIR_NAME :str = "data_ingestion"
DATA_INGESTION__FEATURE_STORE_DIR_NAME :str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME :str = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO :float = 0.2


"Defining variables required for Data validation"
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR_NAME : str = "validated"
DATA_VALIDATION_INVALID_DIR_NAME : str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME : str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILENAME : str = "report.yaml"


"Defining variables required for Data transformation"
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME: str = "transformed_object"
DATA_TRANSFORMATION_IMPUTE_PARAMS : dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"
}
DATA_TRANSFORMED_TRAIN_FILENAME: str = "train.npy"
DATA_TRANSFORMED_TEST_FILENAME: str = "test.npy"


"defining variables required for Model Training"
MODEL_TRAINER_DIR_NAME : str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME : str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILENAME : str = "model.pkl"
MODEL_TRAINER_EXPECTED_MODEL_SCORE : float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD : float = 0.05