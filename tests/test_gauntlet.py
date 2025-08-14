import sys
import pytest
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractor import extract_invoice_data, InvoiceParser

# Define the root of the project to build absolute paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PDFS_DIR = PROJECT_ROOT / "input_pdfs"

@pytest.mark.asyncio
async def test_scanned_pdf_triggers_smart_lane(caplog):
    """Test that a known scanned PDF triggers the Smart Lane."""
    caplog.set_level(logging.INFO)
    
    # This file is known to be a scanned receipt that fails initial parsing
    pdf_path = str(INPUT_PDFS_DIR / "Гориво 29.03.2025 2.pdf")
    
    # Instantiate the parser
    invoice_parser = InvoiceParser()
    
    # Run the extraction process
    data = await extract_invoice_data(pdf_path, enable_ocr=True, invoice_parser=invoice_parser)
    
    # Check that the log indicates a fallback to the Smart Lane
    assert "Executing Smart Lane (OCR + LLM)." in caplog.text
    # Check that the placeholder data from the smart lane is returned
    assert data["invoice_number"] == "LLM_PLACEHOLDER"

@pytest.mark.asyncio
async def test_good_pdf_uses_fast_lane(caplog):
    """Test that a known good, text-based PDF uses the Fast Lane successfully."""
    caplog.set_level(logging.INFO)
    
    # This file is a standard text-based PDF that should be parsed by patterns
    pdf_path = str(INPUT_PDFS_DIR / "Invoice 0000000571 Vukov Development Services LTD.pdf")
    
    invoice_parser = InvoiceParser()
    
    data = await extract_invoice_data(pdf_path, enable_ocr=True, invoice_parser=invoice_parser)
    
    # Check that the log indicates Fast Lane was used and NO fallback occurred
    assert "Attempting Fast Lane (pattern-based) parsing." in caplog.text
    assert "Executing Smart Lane (OCR + LLM)." not in caplog.text
    # Check for some actual data that should have been extracted
    assert data["invoice_number"] == "0000000571"
    assert data["total"] == 300.0

@pytest.mark.asyncio
async def test_poor_match_pdf_falls_back_to_smart_lane(caplog):
    """Test that a text-based PDF that doesn't match patterns well falls back to the Smart Lane."""
    caplog.set_level(logging.INFO)
    
    # This file is text-based but has a layout that the patterns struggle with
    pdf_path = str(INPUT_PDFS_DIR / "Ф-ра 0000000593 Вуков Дивелъпмънт Сървисис ЕООД.pdf")
    
    invoice_parser = InvoiceParser()
    
    data = await extract_invoice_data(pdf_path, enable_ocr=True, invoice_parser=invoice_parser)
    
    # Check that the log shows the fallback
    assert "Fast Lane parsing yielded incomplete data. Falling back to Smart Lane." in caplog.text
    assert "Executing Smart Lane (OCR + LLM)." in caplog.text
    # Check that the placeholder data from the smart lane is returned
    assert data["invoice_number"] == "LLM_PLACEHOLDER"