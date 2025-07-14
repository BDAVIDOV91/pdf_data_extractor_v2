from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    input_dir: Path = Field("input_pdfs", description="Directory containing PDF invoices.")
    output_dir: Path = Field("output_reports", description="Directory to save reports, CSV, and Excel files.")
    log_file: Path = Field("logs/extraction.log", description="Path to the log file.")
    enable_ocr: bool = Field(True, description="Enable OCR for scanned PDFs.")
    tesseract_cmd: str = Field("/usr/bin/tesseract", description="Path to the Tesseract executable.")
    database_path: Path = Field("invoices.db", description="Path to the SQLite database file.")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
