import time
import logging
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from extractor import extract_invoice_data
from report_generator import ReportGenerator
from validator import Validator
from database import DatabaseManager
from config import settings

class PDFHandler(FileSystemEventHandler):
    """Handles file system events, specifically for new PDF files."""
    def __init__(self, db: bool):
        """Initializes the PDFHandler.

        Args:
            db (bool): Flag to enable/disable database storage.
        """
        self.db = db
        self.report_generator = ReportGenerator(settings.output_dir)

    def on_created(self, event: FileSystemEvent) -> None:
        """Called when a file or directory is created.

        Args:
            event (FileSystemEvent): The event object representing the creation.
        """
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logging.info(f"Detected new PDF: {event.src_path}")
            # Run the async processing in a new event loop
            asyncio.run(self.process_pdf(event.src_path))

    async def process_pdf(self, pdf_path: str) -> None:
        """Processes a newly detected PDF file asynchronously.

        Args:
            pdf_path (str): The path to the PDF file to process.
        """
        try:
            # This logic mirrors the process_pdf function in main.py
            extracted_data = await extract_invoice_data(pdf_path)
            if not extracted_data:
                logging.warning(f"No data extracted from {pdf_path}.")
                return

            validated_data = Validator.validate_and_normalize_data(extracted_data)
            self.report_generator.generate_report(validated_data)
            logging.info(f"Successfully processed and reported for {pdf_path}")

            if self.db:
                db_manager = None
                try:
                    db_manager = DatabaseManager(settings.database_url)
                    db_manager.insert_invoice(validated_data)
                    logging.info(f"Successfully saved data for {pdf_path} to the database.")
                except Exception as e:
                    logging.error(f"Database error for {pdf_path}: {e}")
                finally:
                    if db_manager:
                        db_manager.close()

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")

def start_watcher(db: bool) -> None:
    """Starts the file system watcher.

    Args:
        db (bool): Flag to enable/disable database storage.
    """
    logging.info(f"Starting PDF watcher on {settings.input_dir}")
    event_handler = PDFHandler(db)
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