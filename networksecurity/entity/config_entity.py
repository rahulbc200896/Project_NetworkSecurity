import os
from networksecurity.constants import training_pipeline
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_path = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION__FEATURE_STORE_DIR_NAME,training_pipeline.FILENAME)
        self.train_file_path: str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME,training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path: str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME,training_pipeline.TEST_FILE_NAME)

        self.train_test_split_ratio : float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        #self.data_base_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME
        #self.data_collection_name:str = training_pipeline.DATA_INGESTION_COLLECTION_NAME



class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.data_validation_valid_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_VALID_DIR_NAME)
        self.valid_train_path: str = os.path.join(self.data_validation_valid_dir,training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_path: str = os.path.join(self.data_validation_valid_dir,training_pipeline.TEST_FILE_NAME)
        self.data_validation_invalid_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR_NAME)
        self.invalid_train_path: str = os.path.join(self.data_validation_invalid_dir,training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_path: str = os.path.join(self.data_validation_invalid_dir,training_pipeline.TEST_FILE_NAME)
        self.data_validation_drift_report_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILENAME)

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.data_tranformation_tranformed_trained_data: str = os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME,training_pipeline.DATA_TRANSFORMED_TRAIN_FILENAME)
        self.data_tranformation_tranformed_test_data: str = os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR_NAME,training_pipeline.DATA_TRANSFORMED_TEST_FILENAME)
        self.data_transformed_object_data: str = os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR_NAME,training_pipeline.PREPROCESSOR_OBJ_FILE)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.model_trainer_trained_model_object_file_path: str = os.path.join(self.model_trainer_dir,training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR_NAME,training_pipeline.MODEL_TRAINER_TRAINED_MODEL_FILENAME)
        self.model_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_MODEL_SCORE
        self.model_overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD