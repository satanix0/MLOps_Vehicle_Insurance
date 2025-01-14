from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """ Defines path where the artifacts will  be stored after data ingestion"""
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str