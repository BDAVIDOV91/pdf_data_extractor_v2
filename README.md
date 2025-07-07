# PDF Data Extractor & Reporter

This project automates extracting text and structured invoice data from PDF files and generates text reports, CSV, and Excel files. It can run as a one-time batch processor or as a continuous service that watches a folder for new invoices.

## Features

- **Batch Processing**: Process all PDFs in a specified input directory at once.
- **Folder Watching**: Continuously monitor a directory for new PDF files and process them automatically.
- **Accurate Data Extraction**:
  - Extracts key information like invoice number, date, client, total amount, and VAT.
  - Supports multiple invoice layouts through configurable YAML patterns.
- **Line Item Extraction**: Parses detailed line items (description and amount) from invoices.
- **Optional OCR Support**: Can use Tesseract to extract text from scanned or image-based PDFs.
- **Data Validation**: Validates extracted data to ensure basic correctness (e.g., numeric amounts).
- **Database Integration**: Stores all extracted and validated invoice data in a SQLite database for persistent storage and querying.
- **Comprehensive Reporting**:
  - Generates individual text reports for each invoice.
  - Exports a consolidated report of all invoices to both CSV and Excel formats.
  - Creates a summary report grouping totals by client.
- **Robust Logging**: Logs all operations, errors, and summaries to a rotating log file for easy debugging and tracking.
- **Parallel Processing**: Utilizes multiple CPU cores to process large batches of PDFs concurrently for improved performance.

## Project Structure

```
.
├── input_pdfs/           # Place your PDF invoices here
├── output_reports/       # Generated reports (TXT, CSV, Excel)
├── logs/                 # Application logs
├── .github/              # CI/CD workflows
│   └── workflows/
│       └── ci.yml
├── config.py             # Application configuration settings
├── database.py           # SQLite database management
├── Dockerfile            # For containerizing the application
├── exceptions.py         # Custom exception classes
├── extractor.py          # Core PDF text and data extraction logic
├── main.py               # Main CLI entry point
├── patterns.yml          # Regex patterns for different invoice layouts
├── report_generator.py   # Logic for creating reports
├── requirements.txt      # Python dependencies
├── validator.py          # Data validation and normalization
├── watcher.py            # Folder watching functionality
└── README.md
```

## Requirements

- Python 3.8 or higher
- Tesseract (for OCR functionality)
- Poppler (for PDF to image conversion in OCR)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pdf-data-extractor
    ```

2.  **Install system dependencies:**

    *   **For OCR functionality (required for scanned PDFs):**
        -   Install `Tesseract-OCR`.
        -   Install `poppler-utils`.

    *   **On Debian/Ubuntu:**
        ```bash
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr poppler-utils
        ```

    *   **On macOS (using Homebrew):**
        ```bash
        brew install tesseract poppler
        ```

3.  **Create a virtual environment and install Python packages:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

The application is run via a command-line interface (CLI).

### 1. One-Time Batch Processing

Place your PDF files in the `input_pdfs` directory and run:

```bash
python main.py run
```

This will process all PDFs, generate reports in `output_reports/`, and log activity to `logs/extraction.log`.

### 2. Continuous Folder Watching

To have the application watch the `input_pdfs` directory and process new files as they are added, use the `--watch` flag:

```bash
python main.py run --watch
```

### 3. Database Integration

To store the extracted data in the SQLite database (`invoices.db`), use the `--db` flag. This can be combined with either batch or watch mode.

```bash
# Batch mode with database
python main.py run --db

# Watch mode with database
python main.py run --watch --db
```

## Configuration

- **Invoice Patterns**: To add support for a new invoice layout, add a new entry with corresponding regex patterns in `patterns.yml`.
- **Application Settings**: Environment-specific settings (like directory paths or OCR command paths) can be configured in `config.py` or by creating a `.env` file.

## Future Improvements

- **Web Interface**: Develop a web-based UI (using Flask or FastAPI) for uploading files and viewing results.
- **Advanced Validation**: Implement more sophisticated validation rules (e.g., date formats, checksums).
- **Cloud Storage Integration**: Add support for fetching invoices from cloud storage buckets (e.g., S3, Google Cloud Storage).
