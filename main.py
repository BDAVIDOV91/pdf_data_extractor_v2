import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler

import click

from config import settings
from database import DatabaseManager
from extractor import extract_invoice_data
from report_generator import ReportGenerator
from utils.filesystem import FileSystemUtils, get_pdf_files
from validator import Validator
from watcher import start_watcher


def setup_logging() -> None:
    """Configures the logging for the application."""
    log_dir = settings.log_file.parent
    FileSystemUtils.ensure_directory_exists(log_dir)

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = RotatingFileHandler(
        settings.log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


async def process_pdf(pdf_path: str, db: bool) -> dict | None:
    """
    Processes a single PDF file, validates the data, and optionally stores it in the database.
    Each call to this function when db=True will manage its own database connection.
    """
    try:
        logging.info(f"Processing {os.path.basename(pdf_path)}...")
        extracted_data = await extract_invoice_data(pdf_path)
        if not extracted_data:
            logging.warning(f"No data extracted from {os.path.basename(pdf_path)}.")
            return None

        validated_data = Validator.validate_and_normalize_data(extracted_data)
        logging.info(f"Successfully processed and validated {os.path.basename(pdf_path)}.")

        if db:
            db_manager = None
            try:
                db_manager = DatabaseManager(settings.database_url)
                db_manager.insert_invoice(validated_data)
                logging.info(f"Successfully saved data for {os.path.basename(pdf_path)} to the database.")
            except Exception as e:
                logging.error(f"Database error for {os.path.basename(pdf_path)}: {e}")
                # We still return the data even if DB insert fails, as processing was successful
            finally:
                if db_manager:
                    db_manager.close()
        
        return validated_data

    except Exception as e:
        logging.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return None


@click.group()
def cli():
    """PDF Data Extractor & Reporter"""
    setup_logging()


@cli.command()
@click.option(
    "--watch", is_flag=True, help="Enable continuous folder watching for new PDFs."
)
@click.option("--db", is_flag=True, help="Enable storing results in the database.")
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of parallel processes. Defaults to the number of CPU cores.",
)
def run(watch: bool, db: bool, max_workers: int | None) -> None:
    """Main function to run the PDF data extractor and reporter."""
    asyncio.run(amain(watch, db, max_workers))


async def amain(watch: bool, db: bool, max_workers: int | None) -> None:
    """Main async function to run the PDF data extractor and reporter."""
    logging.info("PDF Data Extractor started.")

    input_dir = settings.input_dir
    output_dir = settings.output_dir
    report_generator = ReportGenerator(output_dir)

    # Database tables are created here to ensure they exist before any processing starts.
    if db:
        db_manager = DatabaseManager(settings.database_url)
        db_manager.create_tables()
        db_manager.close()

    if watch:
        # The watcher logic will also need to be updated to handle its own DB connections.
        # For now, we pass the 'db' flag to it.
        logging.info("Starting watcher mode...")
        start_watcher(db) # Assuming start_watcher is updated to accept the bool flag
    else:
        if not os.path.exists(input_dir):
            logging.error(
                f"Input directory '{input_dir}' not found. Please create it and place PDF invoices inside."
            )
            return

        pdf_files = get_pdf_files(input_dir)[:10] # Process only the first 10 PDF files

        if not pdf_files:
            logging.info(f"No PDF files found in '{input_dir}'.")
            return

        all_validated_data = []
        successful_extractions = 0
        failed_extractions = 0

        if max_workers and max_workers > 1:
            logging.info(f"Processing {len(pdf_files)} PDFs with {max_workers} workers...")
            semaphore = asyncio.Semaphore(max_workers)

            async def sem_process_pdf(pdf_path: str):
                async with semaphore:
                    return await process_pdf(pdf_path, db)

            results = await asyncio.gather(*[sem_process_pdf(pdf_file) for pdf_file in pdf_files])
        else:
            logging.info(f"Processing {len(pdf_files)} PDFs sequentially...")
            results = [await process_pdf(pdf_file, db) for pdf_file in pdf_files]

        for result in results:
            if result:
                all_validated_data.append(result)
                report_generator.generate_report(result)
                successful_extractions += 1
            else:
                failed_extractions += 1

        if all_validated_data:
            report_generator.export_to_csv_excel(all_validated_data)
            report_generator.generate_summary_report(all_validated_data)
            logging.info("All data exported to CSV and Excel.")
        else:
            logging.info("No data to export.")

        logging.info("--- Summary ---")
        logging.info(f"Total PDFs processed: {len(pdf_files)}")
        logging.info(f"Successful extractions: {successful_extractions}")
        logging.info(f"Failed extractions: {failed_extractions}")
        logging.info("-----------------")
        logging.info("PDF Data Extractor finished.")


if __name__ == "__main__":
    cli()