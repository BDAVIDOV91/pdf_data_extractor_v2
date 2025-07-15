import logging

logging.basicConfig(level=logging.DEBUG)
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add the project root to the sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extractor import TextExtractor, InvoiceParser
import PyPDF2
from exceptions import UnsupportedInvoiceFormatError
from utils import FileSystemUtils # Import FileSystemUtils for mocking

import unittest

class TestTextExtractor(unittest.TestCase):

    @patch('extractor.PyPDF2')
    @patch("builtins.open", new_callable=mock_open, read_data="dummy data")
    def test_extract_text_from_pdf_success(self, mock_file, mock_pypdf2):
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello World"
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader

        extractor = TextExtractor()
        text = extractor.extract_text_from_pdf("dummy.pdf")
        self.assertEqual(text, "Hello World")

    @patch('extractor.convert_from_path')
    @patch('extractor.pytesseract')
    @patch('extractor.FileSystemUtils.preprocess_image') # Mock preprocess_image
    @patch("builtins.open", new_callable=mock_open, read_data="dummy data")
    def test_extract_text_from_pdf_ocr_success(self, mock_file, mock_preprocess_image, mock_pytesseract, mock_convert_from_path):
        with patch('extractor.PyPDF2.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = PyPDF2.errors.PdfReadError("Invalid PDF")
            mock_convert_from_path.return_value = [MagicMock()]
            mock_pytesseract.image_to_string.return_value = "OCR Text"
            mock_preprocess_image.return_value = MagicMock() # Ensure preprocess_image returns a mock object

            extractor = TextExtractor(enable_ocr=True)
            text = extractor.extract_text_from_pdf("dummy.pdf")
            self.assertEqual(text, "OCR Text")

class TestInvoiceParser(unittest.TestCase):

    @patch('extractor.yaml')
    def setUp(self, mock_yaml):
        mock_yaml.safe_load.return_value = {
            "patent_and_trademark_institute": {
                "document_type": "patent_invoice",
                "keywords": ["Patent and Trademark Institute"],
                "invoice_number": r"Invoice No: (\d+)",
                "date": r"Date:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Total:\s*(?:[A-Z$€£]{1,3}\s*)?([\d.,]+)(?!\s*Subtotal)",
                "vat": r"VAT:\s*([\d.,]+)",
                "client": r"(?:Bill To|Client):\s*([A-Za-z\s.,-]+?)(?:\s*Description|\n|$)",
                "line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "receipt": {
                "document_type": "receipt",
                "keywords": [
                    "receipt",
                    "cash memo",
                    "sales slip",
                    "thank you",
                    "purchase",
                    "store",
                    "transaction"
                ],
                "merchant_name": r'^(.*?)(?:\n|\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4})',
                "date": r'(?:Date|Datum|Dato|Fecha|Date of Issue|Trans Date|Sale Date)[:\s]*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
                "total": r'(?:Total|Amount Due|Grand Total|Balance Due|Sum|Gesamtbetrag|Total Amount|Total Payable|Net Total|Total Incl. VAT|TOTAL)[:\s]*(?:[A-Z$€£]{1,3}\s*)?([\d.,]+)',
                "vat": r'(?:VAT|Tax|MwSt)[:\s]*([\d.,]+)',
                "transaction_id": r'(?:Trans|Transaction|Auth|Approval)\s*ID[:#]*\s*([A-Z0-9]+)',
                "line_items": r'(?P<quantity>\d+)?\s*(?P<description>.+?)\s+(?P<amount>[\d.,]+)'
            },
            "vukov_development_services": {
                "document_type": "vukov_invoice",
                "keywords": [
                    "Vukov Development Services",
                    "Вуков Дивелъпмънт Сървисис ЕООД"
                ],
                "invoice_number": r"(\d+)\s*Invoice No:",
                "date": r"Date:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Total:\s*([\d.,]+)",
                "vat": r"VAT:\s*([\d.,]+)",
                "client": r"([A-Za-z\s.,-]+?)\s*(?:Ship To:|Bill To:)",
                "line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "etkyusi_eood": {
                "document_type": "etkyusi_invoice",
                "keywords": [
                    "ЕтКюСи ЕООД"
                ],
                "invoice_number": r"Фактура\s*No:(\d+)",
                "date": r"Дата на издаване:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Сума за плащане:\s*([\d.,]+)",
                "vat": r"Начислен ДДС:?([\d.,]+)",
                "client": r"Получател:\s*Име на фирма:\s*([^\n]+)",
                "line_items": r"Description\s*Amount\s*(.*?)(?:Междинна сума|Сума за плащане)"
            },
            "replit": {
                "document_type": "replit_invoice",
                "keywords": [
                    "Replit"
                ],
                "invoice_number": r"Invoice\s*number\s*([A-Z0-9\s\.\-]+)",
                "date": r"Date\s*of\s*issue\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
                "total": r"$([\d\s.,]+)\s*U\s*S\s*D"
                ,"vat": r"VAT\s*(\d+%)?:\s*([\d.,]+)"
                ,"client": r"Replit\n([A-Za-z\s.,-]+)"
                ,"line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "atqc_ltd": {
                "document_type": "atqc_invoice",
                "keywords": [
                    "AtQC Ltd"
                ],
                "invoice_number": r"No:(\d+)",
                "date": r"Date of issue:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Due Amount:\s*([\d.,]+)\s*([A-Z]{3})"
                ,"vat": r"VAT\(0%\):\s*([\d.,]+)"
                ,"client": r"Recipient\nCompany Name:([A-Za-z\s.,-]+)"
                ,"line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "document": {
                "document_type": "generic_document",
                "keywords": [],
                "invoice_number": r"(?:Invoice|Bill|Ref|No|Number)[:\s#]*([A-Z0-9\-]+)",
                "date": r"(?:Date|Datum|Dato|Fecha|Date of Issue)[:\s]*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                "total": r"(?:Total|Amount Due|Grand Total|Balance Due|Sum|Gesamtbetrag|Total Amount|Total Payable|Net Total|Total Incl. VAT)[:\s]*(?:[A-Z$€£]{1,3}\s*)?([\d.,]+)",
                "vat": r"(?:VAT|Tax|MwSt)[:\s]*([\d.,]+)",
                "client": r"(?:Bill To|Customer|Client|Recipient)[:\s]*(.*?)(?:\n|$)",                "line_items": r"(.+?)\s+([\d.,]+)"            }        }
        self.parser = InvoiceParser()
        self.sample_text = ""

    def test_normalize_amount(self):
        self.assertEqual(self.parser.normalize_amount("1,234.56"), 1234.56)
        self.assertEqual(self.parser.normalize_amount("1 234,56"), 1234.56)
        self.assertIsNone(self.parser.normalize_amount("invalid"))

    def test_get_document_type(self):
        self.assertEqual(self.parser.get_document_type("Patent and Trademark Institute"), "patent_and_trademark_institute")
        self.assertEqual(self.parser.get_document_type("Vukov Development Services"), "vukov_development_services")
        self.assertEqual(self.parser.get_document_type("Unknown Company"), "document") # Updated to reflect new fallback behavior
        self.assertEqual(self.parser.get_document_type("This is a sales slip"), "receipt") # Test for new receipt type

    def test_parse_invoice_data(self):
        self.sample_text = """
        Patent and Trademark Institute
        Invoice No: 12345
        Date: 05.07.2025
        Client: Test Client
        Description   Amount
        Item 1        10.00
        Item 2        20.00
        Total: 100.00
        VAT: 20.00
        Currency: USD
        """
        data = self.parser.parse_invoice_data(self.sample_text)
        print(f"DEBUG: Document type identified: {self.parser.get_document_type(self.sample_text)}")
        self.assertEqual(data['invoice_number'], "12345")

        # Test for receipt with transaction_id
        receipt_text = """
        Thank You for your purchase!
        Date: 07/14/2025
        Total: 50.00
        Trans ID: ABC123XYZ
        """
        receipt_data = self.parser.parse_invoice_data(receipt_text)
        self.assertEqual(receipt_data['transaction_id'], "ABC123XYZ")

if __name__ == '__main__':
    unittest.main()