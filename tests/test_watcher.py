import unittest
from unittest.mock import patch, MagicMock
from watcher import PDFHandler

class TestPDFHandler(unittest.TestCase):

    @patch('extractor.extract_invoice_data')
    @patch('watcher.ReportGenerator')
    @patch('watcher.settings')
    def setUp(self, mock_settings, MockReportGenerator, mock_extract_invoice_data):
        mock_settings.enable_ocr = False
        mock_settings.output_dir = "test_output"
        self.handler = PDFHandler(db_manager=MagicMock())
        self.mock_extract_invoice_data = mock_extract_invoice_data
        self.mock_report_generator = MockReportGenerator.return_value

    @patch('watcher.PDFHandler.process_pdf')
    def test_on_created_pdf(self, mock_process_pdf):
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/new_invoice.pdf"
        self.handler.on_created(event)
        mock_process_pdf.assert_called_once_with("/path/to/new_invoice.pdf")

    def test_on_created_non_pdf(self):
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/path/to/image.jpg"
        self.handler.on_created(event)
        # Assert that process_pdf was not called
        # This requires patching process_pdf in the test method itself
        with patch('watcher.PDFHandler.process_pdf') as mock_process_pdf:
            self.handler.on_created(event)
            mock_process_pdf.assert_not_called()

    def test_on_created_directory(self):
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/path/to/new_dir"
        self.handler.on_created(event)
        # Assert that process_pdf was not called
        with patch('watcher.PDFHandler.process_pdf') as mock_process_pdf:
            self.handler.on_created(event)
            mock_process_pdf.assert_not_called()

if __name__ == '__main__':
    unittest.main()