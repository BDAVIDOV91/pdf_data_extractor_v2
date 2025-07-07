import unittest
from unittest.mock import patch, MagicMock, mock_open
from extractor import TextExtractor, InvoiceParser
import PyPDF2
from exceptions import UnsupportedInvoiceFormatError

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
    @patch("builtins.open", new_callable=mock_open, read_data="dummy data")
    def test_extract_text_from_pdf_ocr_success(self, mock_file, mock_pytesseract, mock_convert_from_path):
        with patch('extractor.PyPDF2.PdfReader') as mock_pdf_reader:
            mock_pdf_reader.side_effect = PyPDF2.errors.PdfReadError("Invalid PDF")
            mock_convert_from_path.return_value = [MagicMock()]
            mock_pytesseract.image_to_string.return_value = "OCR Text"

            extractor = TextExtractor(enable_ocr=True)
            text = extractor.extract_text_from_pdf("dummy.pdf")
            self.assertEqual(text, "OCR Text")

class TestInvoiceParser(unittest.TestCase):

    @patch('extractor.yaml')
    def setUp(self, mock_yaml):
        mock_yaml.safe_load.return_value = {
            "patent_and_trademark_institute": {
                "document_type": "patent_invoice",
                "invoice_number": r"(\d+)\s*Invoice No:|Invoice No:\s*(\d+)",
                "date": r"Date:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Total:\s*(?:[A-Z$€£]{1,3}\s*)?([\d.,]+)(?!\s*Subtotal)",
                "vat": r"VAT:\s*([\d.,]+)",
                "client": r"Bill To:\s*([A-Za-z\s.,-]+?)(?:\s*№|\s*Phone:|\s*VAT Number:|\n|Invoice No:|Date:)",
                "line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "vukov_development_services": {
                "document_type": "vukov_invoice",
                "invoice_number": r"(\d+)\s*Invoice No:",
                "date": r"Date:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Total:\s*([\d.,]+)",
                "vat": r"VAT:\s*([\d.,]+)",
                "client": r"([A-Za-z\s.,-]+?)\s*(?:Ship To:|Bill To:)",
                "line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "etkyusi_eood": {
                "document_type": "etkyusi_invoice",
                "invoice_number": r"Фактура\s*No:(\d+)",
                "date": r"Дата на издаване:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Сума за плащане:\s*([\d.,]+)",
                "vat": r"Начислен ДДС:?([\d.,]+)",
                "client": r"Получател:\s*Име на фирма:\s*([^\n]+)",
                "line_items": r"Description\s*Amount\s*(.*?)(?:Междинна сума|Сума за плащане)"
            },
            "replit": {
                "document_type": "replit_invoice",
                "invoice_number": r"Invoice\s*number\s*([A-Z0-9\s\.\-]+)",
                "date": r"Date\s*of\s*issue\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
                "total": r"$([\d\s.,]+)\s*U\s*S\s*D"
                ,"vat": r"VAT\s*(\d+%)?:\s*([\d.,]+)"
                ,"client": r"Replit\n([A-Za-z\s.,-]+)"
                ,"line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            },
            "atqc_ltd": {
                "document_type": "atqc_invoice",
                "invoice_number": r"No:(\d+)",
                "date": r"Date of issue:\s*(\d{2}\.\d{2}\.\d{4})",
                "total": r"Due Amount:\s*([\d.,]+)\s*([A-Z]{3})"
                ,"vat": r"VAT\(0%\):\s*([\d.,]+)"
                ,"client": r"Recipient\nCompany Name:([A-Za-z\s.,-]+)"
                ,"line_items": r"Description\s*Amount\s*(.*?)(?:Subtotal|Total)"
            }
        }
        self.parser = InvoiceParser()
        self.sample_text = ""

    def test_normalize_amount(self):
        self.assertEqual(self.parser.normalize_amount("1,234.56"), 1234.56)
        self.assertEqual(self.parser.normalize_amount("1 234,56"), 1234.56)
        self.assertIsNone(self.parser.normalize_amount("invalid"))

    def test_get_document_type(self):
        self.assertEqual(self.parser.get_document_type("Patent and Trademark Institute"), "patent_and_trademark_institute")
        self.assertEqual(self.parser.get_document_type("Vukov Development Services"), "vukov_development_services")
        with self.assertRaises(UnsupportedInvoiceFormatError):
            self.parser.get_document_type("Unknown Company")

    def test_parse_invoice_data(self):
        self.sample_text = """
        Patent and Trademark Institute
        Invoice No: 12345
        Date: 05.07.2025
        Client: Test Client
        Total: 100.00
        VAT: 20.00
        Currency: USD
        """
        data = self.parser.parse_invoice_data(self.sample_text)
        self.assertEqual(data['invoice_number'], "12345")

if __name__ == '__main__':
    unittest.main()