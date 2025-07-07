import unittest
from unittest.mock import patch, MagicMock
from watcher import PDFHandler

class TestPDFHandler(unittest.TestCase):

    @patch('watcher.TextExtractor')
    @patch('watcher.InvoiceParser')
    @patch('watcher.ReportGenerator')
    @patch('watcher.settings')
    def setUp(self, mock_settings, MockReportGenerator, MockInvoiceParser, MockTextExtractor):
        mock_settings.enable_ocr = False
        mock_settings.output_dir = "test_output"
        self.handler = PDFHandler()
        self.mock_text_extractor = MockTextExtractor.return_value
        self.mock_invoice_parser = MockInvoiceParser.return_value
        self.mock_report_generator = MockReportGenerator.return_value

    @patch('watcher.os.path.exists', return_value=False)
    @patch('watcher.pd')
    def test_process_pdf_success(self, mock_pd, mock_exists):
        self.mock_text_extractor.extract_text_from_pdf.return_value = "dummy text"
        self.mock_invoice_parser.parse_invoice_data.return_value = {"invoice_number": "123"}

        self.handler.process_pdf("test.pdf")

        self.mock_text_extractor.extract_text_from_pdf.assert_called_once_with("test.pdf")
        self.mock_invoice_parser.parse_invoice_data.assert_called_once_with("dummy text")
        self.mock_report_generator.generate_report.assert_called_once_with({"invoice_number": "123"})
        self.mock_report_generator.export_to_csv_excel.assert_called_once()
        self.mock_report_generator.generate_summary_report.assert_called_once()

    @patch('watcher.os.path.exists', return_value=False)
    @patch('watcher.pd')
    def test_process_pdf_no_text(self, mock_pd, mock_exists):
        self.mock_text_extractor.extract_text_from_pdf.return_value = None

        self.handler.process_pdf("test.pdf")

        self.mock_text_extractor.extract_text_from_pdf.assert_called_once_with("test.pdf")
        self.mock_invoice_parser.parse_invoice_data.assert_not_called()
        self.mock_report_generator.generate_report.assert_not_called()

    @patch('watcher.os.path.exists', return_value=False)
    @patch('watcher.pd')
    def test_process_pdf_exception(self, mock_pd, mock_exists):
        self.mock_text_extractor.extract_text_from_pdf.side_effect = Exception("Test Error")

        with self.assertLogs(level='ERROR') as cm:
            self.handler.process_pdf("test.pdf")
            self.assertIn("Error processing test.pdf: Test Error", cm.output[0])

if __name__ == '__main__':
    unittest.main()