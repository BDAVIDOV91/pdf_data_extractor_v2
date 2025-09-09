import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validator import Validator

class TestValidator(unittest.TestCase):

    def test_validate_and_normalize_data(self):
        data = {"invoice_number": "INV001", "date": "2023-01-01", "total": "100.00"}
        result = Validator.validate_and_normalize_data(data)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIn("validation_errors", result)
        self.assertEqual(result["data"]["invoice_number"], "INV001")
        self.assertEqual(result["data"]["date"], "2023-01-01")
        self.assertEqual(result["data"]["total"], "100.00")
        self.assertEqual(len(result["validation_errors"]), 0)

    def test_normalize_date(self):
        self.assertEqual(Validator._normalize_date("01.01.2023"), "2023-01-01")
        self.assertEqual(Validator._normalize_date("2023-01-01"), "2023-01-01")
        self.assertIsNone(Validator._normalize_date("invalid-date"))

if __name__ == '__main__':
    unittest.main()