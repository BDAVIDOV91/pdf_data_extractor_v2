import PyPDF2
import re
import logging
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import yaml

with open('patterns.yml', 'r') as f:
    patterns = yaml.safe_load(f)
from config import settings
from exceptions import TextExtractionError, OCRProcessingError, InvoiceParsingError, UnsupportedInvoiceFormatError

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
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
            if text.strip():
                logging.info(f"Extracted text from PDF: {text[:500]}...")
                return text
        except PyPDF2.errors.PdfReadError as e:
            logging.warning(f"PDF Read Error for {pdf_path}: {e}. Attempting OCR...")
        except Exception as e:
            logging.warning(f"Error extracting text from {pdf_path}: {e}. Attempting OCR...")
        
        if self.enable_ocr:
            logging.info(f"Attempting OCR for {pdf_path}...")
            try:
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    text += pytesseract.image_to_string(image)
                if text.strip():
                    return text
                else:
                    raise TextExtractionError(f"OCR extracted no text from {pdf_path}.")
            except Exception as e:
                raise OCRProcessingError(f"OCR failed for {pdf_path}: {e}")
        
        raise TextExtractionError(f"Could not extract text from {pdf_path}.")

class InvoiceParser:
    """Parses invoice data from extracted text using predefined regex patterns."""
    def __init__(self):
        """Initializes the InvoiceParser with patterns loaded from patterns.yml."""
        self.patterns = patterns

    def normalize_amount(self, amount_str: str) -> float | None:
        """Normalizes an amount string to a float, handling different decimal and thousand separators.

        Args:
            amount_str (str): The amount string to normalize.

        Returns:
            float | None: The normalized amount as a float, or None if normalization fails.
        """
        if not isinstance(amount_str, str):
            return None
        cleaned_amount = amount_str.replace(' ', '')
        last_comma = cleaned_amount.rfind(',')
        last_dot = cleaned_amount.rfind('.')
        if last_comma > last_dot:
            cleaned_amount = cleaned_amount.replace('.', '').replace(',', '.')
        else:
            cleaned_amount = cleaned_amount.replace(',', '')
        try:
            return float(cleaned_amount)
        except ValueError as e:
            logging.warning(f"Normalization failed for amount string '{amount_str}': {e}")
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
        if re.search(r"Patent and Trademark Institute", text, re.IGNORECASE):
            return "patent_and_trademark_institute"
        if re.search(r"Vukov Development Services", text, re.IGNORECASE):
            return "vukov_development_services"
        if re.search(r"ЕтКюСи ЕООД", text, re.IGNORECASE):
            return "etkyusi_eood"
        if re.search(r"R\s*e\s*p\s*l\s*i\s*t", text, re.IGNORECASE):
            return "replit"
        if re.search(r"AtQC Ltd", text, re.IGNORECASE):
            return "atqc_ltd"
        raise UnsupportedInvoiceFormatError("Could not determine document type.")

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
            logging.warning(f"No line item patterns defined for document type: {doc_type}")
            return items

        line_items_pattern = layout_patterns["line_items"]
        
        match = re.search(line_items_pattern, text, re.IGNORECASE | re.DOTALL | re.UNICODE)
        
        if match:
            items_block = match.group(1).strip()
            lines = items_block.split('\n')
            logging.info(f"Found line items block for {doc_type}. Parsing {len(lines)} lines.")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                amount_match = re.search(r'([\d.,]+)\s*$', line)
                amount = None
                description = line

                if amount_match:
                    amount_str = amount_match.group(1)
                    amount = self.normalize_amount(amount_str)
                    description = line[:amount_match.start()].strip()
                
                description = re.sub(r'^\d+\s*\.?\s*|^[a-zA-Z]\s*\.?\s*|\[\d+\]\s*|\[[a-zA-Z]\]\s*|\b(?:Quantity|Qty|Unit|Price|Amount|Total|UR Currency:|nit Price Quantity Unit|otal:|Subtotal|VAT|Grand Total|Сума за плащане|Междинна сума)\b\s*\d*\.?\d*\s*', '', description, flags=re.IGNORECASE).strip()

                if description or amount is not None:
                    items.append({"description": description, "amount": amount})
                    logging.info(f"Parsed line item: Description='{description}', Amount={amount}")
                else:
                    logging.warning(f"Could not parse line item: {line}")

        else:
            logging.warning(f"Line items block not found for document type: {doc_type}. Text block: {text[:500]}...")
            
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
            "currency": "Not found"
        }

        if text is None:
            raise InvoiceParsingError("Cannot parse invoice data: text is None.")

        doc_type = self.get_document_type(text)

        logging.info(f"Identified document type as: {doc_type}")
        layout_patterns = self.patterns.get(doc_type)

        if not layout_patterns:
            raise InvoiceParsingError(f"No patterns found for document type: {doc_type}")

        logging.info(f"Using patterns: {layout_patterns}")

        invoice_number_match = re.search(layout_patterns["invoice_number"], text, re.IGNORECASE | re.UNICODE)
        date_match = re.search(layout_patterns["date"], text, re.IGNORECASE | re.UNICODE)
        total_match = re.search(layout_patterns["total"], text, re.IGNORECASE | re.UNICODE)
        vat_match = re.search(layout_patterns["vat"], text, re.IGNORECASE | re.UNICODE)
        
        client_match = None
        if "client" in layout_patterns:
            client_match = re.search(layout_patterns["client"], text, re.IGNORECASE | re.UNICODE | re.DOTALL)

        if invoice_number_match:
            groups = invoice_number_match.groups()
            inv_num = next((g for g in groups if g is not None), "Not found")
            data["invoice_number"] = inv_num.strip().replace(' ', '')
            logging.info(f"Found invoice number: {data['invoice_number']}")
        else:
            logging.warning("Invoice number not found.")

        if date_match:
            data["date"] = date_match.group(1).strip().replace(' ', '')
            logging.info(f"Found date: {data['date']}")
        else:
            logging.warning("Date not found.")

        if doc_type == "patent_and_trademark_institute":
            all_totals = re.findall(layout_patterns["total"], text, re.IGNORECASE | re.UNICODE)
            if all_totals:
                total_val = all_totals[-1]
                if isinstance(total_val, tuple):
                    total_val = next((item for item in total_val if item), None)
                data["total"] = self.normalize_amount(total_val.strip())
                logging.info(f"Found total (patent): {data['total']}")
            else:
                logging.warning("Total not found for patent invoice.")
        elif total_match:
            data["total"] = self.normalize_amount(total_match.group(1).strip())
            logging.info(f"Found total: {data['total']}")
        else:
            logging.warning("Total not found.")

        if vat_match:
            vat_value = vat_match.group(1)
            if vat_value:
                 data["vat"] = self.normalize_amount(vat_value.strip())
                 logging.info(f"Found VAT: {data['vat']}")
            else:
                data["vat"] = 0.0
                logging.info("VAT found but value is empty, setting to 0.0")
        else:
            logging.warning("VAT not found.")

        if client_match:
            data["client"] = client_match.group(1).strip().replace('\n', ' ')
            logging.info(f"Found client: {data['client']}")
        else:
            logging.warning("Client not found.")

        if total_match and len(total_match.groups()) > 1 and total_match.group(2):
            data["currency"] = total_match.group(2).strip()
        elif data["total"] != "Not found":
            if re.search(r'BGN|лв', text, re.IGNORECASE):
                data["currency"] = "BGN"
            elif re.search(r'EUR|€', text, re.IGNORECASE):
                data["currency"] = "EUR"
            elif re.search(r'USD|\$', text, re.IGNORECASE):
                data["currency"] = "USD"
            else:
                data["currency"] = "EUR"
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