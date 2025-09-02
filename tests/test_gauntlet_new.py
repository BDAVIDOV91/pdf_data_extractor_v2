import sys
import pytest
import asyncio
import logging
from pathlib import Path
import json
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor import extract_invoice_data
from exceptions import InvoiceParsingError

# Define the root of the project to build absolute paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PDFS_DIR = PROJECT_ROOT / "input_pdfs"
TESTS_DIR = PROJECT_ROOT / "tests"

@pytest.mark.asyncio
@patch('extractor.TextExtractor')
async def test_good_pdf_uses_fast_lane(mock_text_extractor_class, caplog):
    """Test that a known good, text-based PDF uses the Fast Lane successfully."""
    caplog.set_level(logging.INFO)
    
    # Configure the mock TextExtractor instance
    mock_text_extractor_class.return_value.extract_text.return_value = "Invoice No: 0000000571\nDate: 2023-01-01\nTotal: 300.00\nClient: Vukov Development Services LTD."
    
    # This file is a standard text-based PDF that should be parsed by patterns
    pdf_path = str(INPUT_PDFS_DIR / "Invoice 0000000571 Vukov Development Services LTD.pdf")
    
    data = await extract_invoice_data(pdf_path)
    
    # Check that the log indicates Fast Lane was used and successful
    assert "Attempting 'Fast Lane' extraction" in caplog.text
    assert "'Fast Lane' extraction successful" in caplog.text
    assert "Switching to 'Smart Lane'" not in caplog.text
    
    # Check for some actual data that should have been extracted
    assert data["invoice_number"] == "0000000571"
    assert data["total"] == "300.00"