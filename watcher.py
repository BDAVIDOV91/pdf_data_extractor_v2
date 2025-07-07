import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from extractor import extract_invoice_data
from report_generator import ReportGenerator
from validator import Validator
from database import DatabaseManager
from config import settings

class PDFHandler(FileSystemEventHandler):
    """Handles file system events, specifically for new PDF files."""
    def __init__(self, db_manager: DatabaseManager | None):
        """Initializes the PDFHandler.

        Args:
            db_manager (DatabaseManager | None): The database manager instance.
        """
        self.report_generator = ReportGenerator(settings.output_dir)
        self.validator = Validator()
        self.db_manager = db_manager

    def on_created(self, event: FileSystemEvent) -> None:
        """Called when a file or directory is created.

        Args:
            event (FileSystemEvent): The event object representing the creation.
        """
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logging.info(f"Detected new PDF: {event.src_path}")
            self.process_pdf(event.src_path)

    def process_pdf(self, pdf_path: str) -> None:
        """Processes a newly detected PDF file.

        Args:
            pdf_path (str): The path to the PDF file to process.
        """
        try:
            extracted_data = extract_invoice_data(pdf_path, enable_ocr=settings.enable_ocr)
            if extracted_data:
                validated_data = self.validator.validate_and_normalize_data(extracted_data)
                self.report_generator.generate_report(validated_data)
                if self.db_manager:
                    self.db_manager.insert_invoice(validated_data)
                logging.info(f"Successfully processed and reported for {pdf_path}")
            else:
                logging.warning(f"No data extracted from {pdf_path}.")
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")

def start_watcher(db_manager: DatabaseManager | None) -> None:
    """Starts the file system watcher.

    Args:
        db_manager (DatabaseManager | None): The database manager instance.
    """
    logging.info(f"Starting PDF watcher on {settings.input_dir}")
    event_handler = PDFHandler(db_manager)
    observer = Observer()
    observer.schedule(event_handler, settings.input_dir, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logging.info("PDF watcher stopped.")
