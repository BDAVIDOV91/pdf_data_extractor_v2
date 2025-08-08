import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    input_dir: Path = Field(
        "input_pdfs", description="Directory containing PDF invoices."
    )
    output_dir: Path = Field(
        "output_reports", description="Directory to save reports, CSV, and Excel files."
    )
    log_file: Path = Field("logs/extraction.log", description="Path to the log file.")
    enable_ocr: bool = Field(True, description="Enable OCR for scanned PDFs.")
    OCR_PREPROCESSING_ENABLED: bool = Field(True, description="Enable image pre-processing for OCR.")
    tesseract_cmd: str = Field(
        "/usr/bin/tesseract", description="Path to the Tesseract executable."
    )
    database_url: str = Field(..., description="URL for the PostgreSQL database.")
    

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.database_url = settings.database_url.strip('"\' ').strip()
os.environ["JIGSAW_API_KEY"] = "sk_7f6b1eae79777501ab1a9bf9b1d0ad520df9bfafa6a077e97d10c4457a339e5f9cf6765844485b57275d8e0a8969c84a23c2449f08eeef14911cacd353ccaabf024j3PwthFGU2HS7eIVXI"
