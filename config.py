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
    jigsaw_api_key: str = Field(..., alias="JIGSAW_API_KEY", description="API key for JigsawStack.")

    llama_cloud_api_key: str = Field(..., alias="LLAMA_CLOUD_API_KEY", description="API key for Llama Cloud.")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.database_url = settings.database_url.strip('"\' ').strip()
