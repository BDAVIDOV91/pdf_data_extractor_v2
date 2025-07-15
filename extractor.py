import logging
import re
from multiprocessing import Process, Queue
from queue import Empty

import PyPDF2
import pytesseract
import yaml
from pdf2image import convert_from_path
from PIL import Image

from utils import FileSystemUtils  # Import FileSystemUtils


from config import settings
import yaml
from exceptions import (
    InvoiceParsingError,
    OCRProcessingError,
    TextExtractionError,
    UnsupportedInvoiceFormatError,
)


class TextExtractor:
    """Handles text extraction from PDF files, with optional OCR capabilities."""

    def __init__(self, enable_ocr: bool = False):
        """Initializes the TextExtractor.

        Args:
            enable_ocr (bool): Whether to enable OCR for scanned PDFs.
        """
        self.enable_ocr = enable_ocr
        if self.enable_ocr:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    def _ocr_process(self, pdf_path: str, q: Queue):
        text = ""
        try:
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Pre-process image before passing to Tesseract
                processed_image = FileSystemUtils.preprocess_image(image)
                text += pytesseract.image_to_string(processed_image)
            q.put(text)
        except Exception as e:
            logging.error(
                f"OCR processing failed for {pdf_path} on image {i+1 if 'i' in locals() else 'N/A'}: {e}",
                exc_info=True,
            )
            q.put(f"OCR_ERROR: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file. Tries direct extraction first, then falls back to OCR if enabled.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The extracted text content of the PDF.

        Raises:
            TextExtractionError: If text extraction fails (either direct or OCR).
            OCRProcessingError: If an error occurs during OCR processing.
        """
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
            if text.strip():
                logging.info(f"Extracted text from PDF: {text[:500]}...")
                return text
        except PyPDF2.errors.PdfReadError as e:
            logging.warning(f"PDF Read Error for {pdf_path}: {e}. Attempting OCR...")
        except Exception as e:
            logging.warning(
                f"Error extracting text from {pdf_path}: {e}. Attempting OCR..."
            )

        if self.enable_ocr:
            logging.info(f"Attempting OCR for {pdf_path}...")
            q = Queue()
            p = Process(target=self._ocr_process, args=(pdf_path, q))
            p.start()
            try:
                # Wait for 180 seconds for OCR to complete
                text = q.get(timeout=180)
                if text.strip():
                    return text
                else:
                    raise TextExtractionError(f"OCR extracted no text from {pdf_path}.")
            except Empty:
                p.terminate()
                p.join()
                raise OCRProcessingError(f"OCR process timed out for {pdf_path}.")
            except Exception as e:
                p.terminate()
                p.join()
                raise OCRProcessingError(f"OCR failed for {pdf_path}: {e}")
            finally:
                p.join()  # Ensure the process is cleaned up

        raise TextExtractionError(f"Could not extract text from {pdf_path}.")


class InvoiceParser:
    """Parses invoice data from extracted text using predefined regex patterns."""

    def __init__(self):
        """Initializes the InvoiceParser with patterns loaded from patterns.yml."""
        with open("patterns.yml", "r") as f:
            self.patterns = yaml.safe_load(f)

    def normalize_amount(self, amount_str: str) -> float | None:
        """Normalizes an amount string to a float, handling different decimal and thousand separators.

        Args:
            amount_str (str): The amount string to normalize.

        Returns:
            float | None: The normalized amount as a float, or None if normalization fails.
        """
        if not isinstance(amount_str, str):
            return None
        cleaned_amount = amount_str.replace(" ", "")
        last_comma = cleaned_amount.rfind(",")
        last_dot = cleaned_amount.rfind(".")
        if last_comma > last_dot:
            cleaned_amount = cleaned_amount.replace(".", "").replace(",", ".")
        else:
            cleaned_amount = cleaned_amount.replace(",", "")
        try:
            return float(cleaned_amount)
        except ValueError as e:
            logging.warning(
                f"Normalization failed for amount string '{amount_str}': {e}"
            )
            return None

    def get_document_type(self, text: str) -> str:
        """Determines the type of document based on keywords in the text.

        Args:
            text (str): The text content of the document.

        Returns:
            str: The identified document type (e.g., "patent_and_trademark_institute").

        Raises:
            UnsupportedInvoiceFormatError: If the document type cannot be determined.
        """
        # Iterate through document types in the order they appear in patterns.yml
        for doc_type, patterns in self.patterns.items():
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if re.search(keyword, text, re.IGNORECASE):
                        return doc_type
        
        # Fallback mechanism: if no specific document type is identified, return "document"
        return "document"

    def parse_line_items(self, text: str, doc_type: str) -> list:
        """Parses line items from the invoice text based on document-specific patterns.

        Args:
            text (str): The text content of the invoice.
            doc_type (str): The identified document type.

        Returns:
            list: A list of dictionaries, each representing a line item with 'description' and 'amount'.
        """
        items = []
        layout_patterns = self.patterns.get(doc_type)

        if not layout_patterns or "line_items" not in layout_patterns:
            logging.warning(
                f"No line item patterns defined for document type: {doc_type}"
            )
            return items

        line_items_pattern = layout_patterns["line_items"]

        # Use finditer to get all matches with named groups
        for match in re.finditer(line_items_pattern, text, re.IGNORECASE | re.DOTALL | re.UNICODE):
            item_data = match.groupdict()
            quantity = item_data.get("quantity")
            description = item_data.get("description", "").strip()
            amount_str = item_data.get("amount")
            amount = self.normalize_amount(amount_str)

            if description or amount is not None:
                item = {"description": description, "amount": amount}
                if quantity:
                    item["quantity"] = int(quantity) # Convert quantity to int if present
                items.append(item)
                logging.info(
                    f"Parsed line item: Description='{description}', Amount={amount}, Quantity={quantity}"
                )
            else:
                logging.warning(f"Could not parse line item from match: {match.group(0)}")

        if not items: # If no items were found after iterating through all matches
            logging.warning(
                f"No line items found for document type: {doc_type}. Text block: {text[:500]}..."
            )

        return items

    def parse_invoice_data(self, text: str) -> dict:
        """Parses all relevant invoice data from the extracted text.

        Args:
            text (str): The text content of the invoice.

        Returns:
            dict: A dictionary containing the extracted invoice data.

        Raises:
            InvoiceParsingError: If the input text is None or no patterns are found for the document type.
        """
        data = {
            "invoice_number": "Not found",
            "date": "Not found",
            "client": "Not found",
            "line_items": [],
            "total": "Not found",
            "vat": "Not found",
            "currency": "Not found",
            "transaction_id": "Not found"
        }

        if text is None:
            raise InvoiceParsingError("Cannot parse invoice data: text is None.")

        doc_type = self.get_document_type(text)

        logging.info(f"Identified document type as: {doc_type}")
        logging.debug(f"DEBUG: All patterns: {self.patterns}")

        # Use specific patterns if document type is recognized, otherwise use generic 'document' patterns
        if doc_type in self.patterns:
            layout_patterns = self.patterns.get(doc_type)
        else:
            layout_patterns = self.patterns.get(
                "document"
            )  # Fallback to generic document patterns
            logging.warning(
                f"No specific patterns found for document type: {doc_type}. Using generic document patterns."
            )

        if (
            not layout_patterns
        ):  # Should not happen if 'document' patterns are always present
            raise InvoiceParsingError(
                f"No patterns found for document type: {doc_type} and no generic fallback available."
            )

        logging.info(f"Using patterns: {layout_patterns}")
        logging.debug(f"DEBUG: Layout patterns being used: {layout_patterns}")
        print(f"DEBUG: Layout patterns being used: {layout_patterns}")

        fields_to_extract = ["invoice_number", "date", "client", "total", "vat"]

        for field in fields_to_extract:
            if field in layout_patterns:
                match = re.search(
                    layout_patterns[field],
                    text,
                    re.IGNORECASE
                    | re.UNICODE
                    | (re.DOTALL if field == "client" else 0),
                )
                if match:
                    logging.debug(f"DEBUG: Field: {field}, Match: {match}")
                    extracted_value = match.group(1).strip()
                    if field in ["total", "vat"]:
                        data[field] = self.normalize_amount(extracted_value)
                        if field == "vat" and data[field] is None:
                            data[field] = (
                                0.0  # Set VAT to 0.0 if found but value is empty
                            )
                    else:
                        data[field] = (
                            extracted_value.replace("\n", " ")
                            if field == "client"
                            else extracted_value.replace(" ", "")
                        )
                    logging.info(f"Found {field}: {data[field]}")
                else:
                    logging.warning(
                        f"{field.replace('_', ' ').capitalize()} not found."
                    )

        # Extract transaction_id for receipts
        if doc_type == "receipt" and "transaction_id" in layout_patterns:
            match = re.search(layout_patterns["transaction_id"], text, re.IGNORECASE)
            if match:
                data["transaction_id"] = match.group(1).strip()
                logging.info(f"Found transaction_id: {data['transaction_id']}")
            else:
                logging.warning("Transaction ID not found for receipt.")

        # Special handling for patent_and_trademark_institute total (multiple totals)
        if doc_type == "patent_and_trademark_institute" and "total" in layout_patterns:
            all_totals = re.findall(
                layout_patterns["total"], text, re.IGNORECASE | re.UNICODE
            )
            if all_totals:
                total_val = all_totals[-1]
                if isinstance(total_val, tuple):
                    total_val = next((item for item in total_val if item), None)
                data["total"] = self.normalize_amount(total_val.strip())
                logging.info(f"Found total (patent): {data['total']}")
            else:
                logging.warning("Total not found for patent invoice.")

        # Currency extraction (can be generalized further if patterns become more complex)
        if data["total"] != "Not found":
            if re.search(r"BGN|лв", text, re.IGNORECASE):
                data["currency"] = "BGN"
            elif re.search(r"EUR|€", text, re.IGNORECASE):
                data["currency"] = "EUR"

            else:
                data["currency"] = "EUR"  # Default to EUR if no specific currency found
        else:
            logging.warning("Currency not determined as total was not found.")

        logging.info(f"Found currency: {data['currency']}")

        data["line_items"] = self.parse_line_items(text, doc_type)
        return data


def extract_invoice_data(pdf_path: str, enable_ocr: bool = False) -> dict:
    """Extracts and parses invoice data from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        enable_ocr (bool): Whether to enable OCR for scanned PDFs.

    Returns:
        dict: A dictionary containing the extracted invoice data.

    Raises:
        TextExtractionError: If text extraction fails.
        OCRProcessingError: If OCR processing fails.
        InvoiceParsingError: If invoice parsing fails.
        UnsupportedInvoiceFormatError: If the invoice format is not supported.
    """
    text_extractor = TextExtractor(enable_ocr=enable_ocr)
    text = text_extractor.extract_text_from_pdf(pdf_path)
    invoice_parser = InvoiceParser()
    return invoice_parser.parse_invoice_data(text)
