import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from report_generator import ReportGenerator

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.output_dir = "test_output"
        self.report_generator = ReportGenerator(self.output_dir)
        self.sample_data = {
            "data": {
                "invoice_number": "12345",
                "date": "2025-07-05",
                "client": "Test Client",
                "total": 100.00,
                "vat": 20.00,
                "currency": "USD",
                "line_items": [
                    {"description": "Item 1", "amount": 50.00},
                    {"description": "Item 2", "amount": 50.00}
                ]
            },
            "validation_errors": []
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_generate_report(self, mock_exists, mock_file):
        self.report_generator.generate_report(self.sample_data)
        mock_file.assert_called_once_with(f"{self.output_dir}/invoice_report_12345.txt", 'w')
        handle = mock_file()
        handle.write.assert_any_call("--- Invoice Data Report ---\n")
        handle.write.assert_any_call("Invoice Number: 12345\n")

    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame.to_excel")
    @patch("os.path.exists", return_value=True)
    def test_export_to_csv_excel(self, mock_exists, mock_to_excel, mock_to_csv):
        self.report_generator.export_to_csv_excel([self.sample_data])
        mock_to_csv.assert_called_once_with(f"{self.output_dir}/all_invoices.csv", index=False)
        mock_to_excel.assert_called_once_with(f"{self.output_dir}/all_invoices.xlsx", index=False)

    @patch("pandas.DataFrame.to_csv")
    @patch("os.path.exists", return_value=True)
    def test_generate_summary_report(self, mock_exists, mock_to_csv):
        self.report_generator.generate_summary_report([self.sample_data])
        mock_to_csv.assert_called_once_with(f"{self.output_dir}/summary_report.csv", index=False)

if __name__ == '__main__':
    unittest.main()