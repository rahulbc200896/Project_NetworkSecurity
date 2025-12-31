from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_trained_filepath:str
    valid_test_filepath:str
    invalid_trained_filepath:str
    invalid_test_filepath:str
    data_drift_filepath:str

@dataclass
class DataTransformationArtifact:
    transformed_trained_filepath:str
    transformed_test_filepath:str
    transformed_object_filepath:str

@dataclass
class ClassificationMetric:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_filepath: str
    trained_metric_path: ClassificationMetric
    test_metric_path: ClassificationMetric