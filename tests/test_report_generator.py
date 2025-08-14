import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from report_generator import ReportGenerator

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        self.output_dir = "test_output"
        self.report_generator = ReportGenerator(self.output_dir)
        self.sample_data_full = {
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
        self.sample_data_missing = {
            "data": {
                "invoice_number": "54321",
                "date": None,
                "client": "Another Client",
                "total": 200.00,
                "vat": None,
                "currency": "EUR",
                "line_items": []
            },
            "validation_errors": ["Missing required field: date", "Missing required field: vat"]
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_generate_report_full_data(self, mock_exists, mock_file):
        self.report_generator.generate_report(self.sample_data_full)
        handle = mock_file()
        handle.write.assert_any_call("Invoice Number: 12345\n")
        handle.write.assert_any_call("Total: 100.0\n")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_generate_report_missing_data(self, mock_exists, mock_file):
        self.report_generator.generate_report(self.sample_data_missing)
        handle = mock_file()
        handle.write.assert_any_call("Date: Not Found\n")
        handle.write.assert_any_call("Vat: Not Found\n")

    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame.to_excel")
    @patch("os.path.exists", return_value=True)
    def test_export_to_csv_excel_with_missing_data(self, mock_exists, mock_to_excel, mock_to_csv):
        with patch('pandas.DataFrame') as mock_df:
            self.report_generator.export_to_csv_excel([self.sample_data_full, self.sample_data_missing])
            
            # Check the data passed to the DataFrame constructor
            passed_data = mock_df.call_args[0][0]
            self.assertEqual(passed_data[1]['date'], "Not Found")
            self.assertEqual(passed_data[1]['vat'], "Not Found")

    @patch("pandas.DataFrame.to_csv")
    @patch("os.path.exists", return_value=True)
    def test_generate_summary_report(self, mock_exists, mock_to_csv):
        self.report_generator.generate_summary_report([self.sample_data_full, self.sample_data_missing])
        mock_to_csv.assert_called_once_with(f"{self.output_dir}/summary_report.csv", index=False)

if __name__ == '__main__':
    unittest.main()