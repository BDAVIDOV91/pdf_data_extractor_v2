import base64
import logging
import os
import re
from multiprocessing import Process, Queue
from queue import Empty

import httpx
import PyPDF2
import pytesseract
import yaml
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

from config import settings
from exceptions import (InvoiceParsingError, OCRProcessingError,
                        TextExtractionError, UnsupportedInvoiceFormatError)
from utils import FileSystemUtils, pymupdf_extract
from parsemypdf.utils.pdf_to_image import PDFToJPGConverter


class TextExtractor:
    """Handles text extraction from PDF files, using a tiered approach."""

    def __init__(self, enable_ocr: bool = False):
        """Initializes the TextExtractor.

        Args:
            enable_ocr (bool): Whether to enable OCR for scanned PDFs.
        """
        self.enable_ocr = enable_ocr
        self.jigsaw_api_key = os.environ.get("JIGSAW_API_KEY")


    async def extract_text(self, pdf_path: str) -> str:
        """Extracts text from a PDF using a tiered approach.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The extracted text.
        """
        # Tier 1: Try local PyMuPDF
        try:
            text = pymupdf_extract(pdf_path)
            if text and text.strip():
                logging.info("Extracted text using local PyMuPDF.")
                return text
        except Exception as e:
            logging.warning(f"Local PyMuPDF failed: {e}. Falling back to local OCR.")

        # Tier 2: Local OCR with image pre-processing
        if self.enable_ocr:
            logging.info("Attempting local OCR with image pre-processing.")
            try:
                converter = PDFToJPGConverter()
                images = converter.convert_pdf(pdf_path)
                full_text = []
                for img in images:
                    # Convert PIL Image to OpenCV format
                    open_cv_image = np.array(img)
                    # Convert RGB to BGR
                    open_cv_image = open_cv_image[:, :, ::-1].copy()

                    # Pre-processing: Grayscale, Deskew, and Adaptive Thresholding
                    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

                    # Deskewing (simple example, can be improved)
                    coords = np.column_stack(np.where(gray > 0))
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                    # Adaptive Thresholding
                    thresh = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

                    # Perform OCR
                    text = pytesseract.image_to_string(thresh)
                    full_text.append(text)
                
                if full_text:
                    logging.info("Extracted text using local OCR.")
                    return "\n".join(full_text)
                else:
                    raise OCRProcessingError("Local OCR failed to extract text.")
            except Exception as e:
                logging.error(f"Error during local OCR processing: {e}")
                raise OCRProcessingError(f"Local OCR processing failed: {e}")
        
        raise TextExtractionError(f"Could not extract text from {pdf_path} using any method.")


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
        """Determines the document type by scoring pattern matches."""
        scores = {}
        for doc_type, patterns in self.patterns.items():
            if doc_type == 'document': continue  # Skip generic document for scoring
            
            score = 0
            # Score based on keyword matches
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if re.search(keyword, text, re.IGNORECASE):
                        score += 1
            
            # Score based on other pattern matches (invoice_number, date, etc.)
            for field, pattern in patterns.items():
                if field not in ["keywords", "document_type", "line_items", "line_items_block"]:
                    if re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.UNICODE):
                        score += 2  # Higher weight for specific field matches

            if score > 0:
                scores[doc_type] = score

        if not scores:
            return "document"  # Fallback to generic

        # Return the document type with the highest score
        best_match = max(scores, key=scores.get)
        return best_match

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

        layout_patterns = self.patterns.get(doc_type, {})
        line_items_block_pattern = layout_patterns.get("line_items_block")
        line_items_pattern = layout_patterns.get("line_items")

        if not line_items_pattern:
            logging.warning(f"No line item patterns defined for document type: {doc_type}")
            return items

        # If a block pattern is defined, search within that block
        if line_items_block_pattern:
            block_match = re.search(line_items_block_pattern, text, re.IGNORECASE | re.DOTALL | re.UNICODE)
            if block_match:
                text_block = block_match.group(1)
                logging.info(f"Found line items block for {doc_type}.")
            else:
                logging.warning(f"Line items block not found for {doc_type}.")
                text_block = text  # Fallback to full text
        else:
            text_block = text

        logging.info(f"Attempting to find line items with pattern: {line_items_pattern}")

        # Use finditer to get all matches with named groups
        found_matches = False
        for match in re.finditer(line_items_pattern, text_block, re.IGNORECASE | re.DOTALL | re.UNICODE):
            found_matches = True
            item_data = match.groupdict()
            description = item_data.get("description", "").strip()
            amount_str = item_data.get("amount")
            amount = self.normalize_amount(amount_str)
            quantity = item_data.get("quantity")

            if description or amount is not None:
                item = {"description": description, "amount": amount}
                if quantity:
                    try:
                        item["quantity"] = int(quantity)
                    except (ValueError, TypeError):
                        item["quantity"] = None
                items.append(item)
                logging.info(
                    f"Parsed line item: Description='{description}', Amount={amount}, Quantity={item.get('quantity')}"
                )

        if not found_matches:
            logging.warning(
                f"No line items found for document type: {doc_type} using pattern: {line_items_pattern}."
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
                pattern = layout_patterns[field]
                logging.info(f"Attempting to find '{field}' with pattern: {pattern}")
                match = re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                    | re.UNICODE
                    | (re.DOTALL if field == "client" else 0),
                )
                if match:
                    extracted_value = match.group(1).strip()
                    logging.info(f"SUCCESS: Found raw value for '{field}': '{extracted_value}'")
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
                    logging.info(f"Stored value for '{field}': {data[field]}")
                else:
                    logging.warning(f"'{field.replace('_', ' ').capitalize()}' not found with pattern: {pattern}")

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


async def extract_invoice_data(pdf_path: str, enable_ocr: bool = False) -> dict:
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
    text = await text_extractor.extract_text(pdf_path)
    invoice_parser = InvoiceParser()
    return invoice_parser.parse_invoice_data(text)
