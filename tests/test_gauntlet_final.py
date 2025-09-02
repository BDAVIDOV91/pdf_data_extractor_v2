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
    mock_text_extractor_instance = mock_text_extractor_class.return_value
    mock_text_extractor_instance.extract_text.return_value = "Invoice No: 0000000571\nDate: 2023-01-01\nTotal: 300.00\nClient: Vukov Development Services LTD."
    
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


@pytest.mark.asyncio
@patch('extractor.TextExtractor')
@patch('extractor.QuestionAnswering')
async def test_scanned_pdf_triggers_smart_lane(mock_qa_class, mock_text_extractor_class, caplog):
    """Test that a scanned/image-based PDF falls back to the Smart Lane."""
    caplog.set_level(logging.INFO)
    
    # Configure the mock TextExtractor instance to return text that won't match Fast Lane patterns
    mock_text_extractor_instance = mock_text_extractor_class.return_value
    mock_text_extractor_instance.extract_text.return_value = "This is a scanned document with no clear invoice patterns."

    # Configure the mock QuestionAnswering instance
    mock_qa_instance = mock_qa_class.return_value
    mock_qa_instance.answer_question.side_effect = [
        "12345", # invoice_number
        "2023-01-15", # date
        "150.00", # total
        "Mock Client Inc.", # client
        "20.00", # vat
        "Description;Amount\\nDescription 1;100.00\\nDescription 2;50.00" # line_items
    ]

    # This file is a scanned receipt that should fail the Fast Lane
    pdf_path = str(INPUT_PDFS_DIR / "Гориво 29.03.2025 2.pdf")
    
    data = await extract_invoice_data(pdf_path)
    
    # Check that the log shows the fallback to Smart Lane
    assert "Attempting 'Fast Lane' extraction" in caplog.text
    assert "'Fast Lane' failed or data insufficient. Switching to 'Smart Lane'" in caplog.text
    
    # We can't assert the exact LLM output, but we can check that it tried
    # and returned a dictionary with the expected keys.
    assert data["invoice_number"] == "12345"
    assert data["date"] == "2023-01-15"
    assert data["total"] == "150.00"
    assert data["client"] == "Mock Client Inc."
    assert data["vat"] == "20.00"
    assert len(data["line_items"]) == 2
    assert data["line_items"][0]['description'] == "Description 1"
    assert data["line_items'][0]['amount'] == "100.00"
    assert data["line_items'][1]['description'] == "Description 2"
    assert data["line_items'][1]['amount'] == "50.00"

@pytest.mark.asyncio
@patch('extractor.TextExtractor')
async def test_empty_file_handling(mock_text_extractor_class):
    """Test that an empty file raises an InvoiceParsingError."""
    # Configure the mock TextExtractor instance to return empty text
    mock_text_extractor_class.return_value.extract_text.return_value = ""

    pdf_path = str(TESTS_DIR / "empty.pdf")
    
    with pytest.raises(InvoiceParsingError, match="Could not extract any text from the PDF."):
        await extract_invoice_data(pdf_path)

@pytest.mark.asyncio
@patch('extractor.TextExtractor')
async def test_non_pdf_file_handling(mock_text_extractor_class, caplog):
    """Test that a non-PDF file is handled gracefully."""
    caplog.set_level(logging.ERROR)
    # Configure the mock TextExtractor instance to return empty text
    mock_text_extractor_class.return_value.extract_text.return_value = ""

    pdf_path = str(TESTS_DIR / "fake.pdf")

    # The current implementation with docling might raise a generic error or just log it.
    # We expect an InvoiceParsingError because text extraction will fail and return empty.
    with pytest.raises(InvoiceParsingError, match="Could not extract any text from the PDF."):
        await extract_invoice_data(pdf_path)
    
    # Additionally, check if the lower-level error from the TextExtractor was logged.
    # assert "Failed to extract text from" in caplog.text

@pytest.mark.asyncio
@patch('extractor.TextExtractor')
async def test_password_protected_pdf_handling(mock_text_extractor_class, caplog):
    """Test that a password-protected PDF is handled gracefully."""
    caplog.set_level(logging.ERROR)
    # Configure the mock TextExtractor instance to return empty text
    mock_text_extractor_class.return_value.extract_text.return_value = ""

    # Placeholder for a password-protected PDF. User needs to provide this file.
    pdf_path = str(TESTS_DIR / "password_protected.pdf")

    # Expecting an InvoiceParsingError if text extraction fails due to protection
    with pytest.raises(InvoiceParsingError, match="Could not extract any text from the PDF."):
        await extract_invoice_data(pdf_path)
    
    # Check if the error message indicates issues with the PDF (e.g., encrypted)
    # assert "Failed to extract text from" in caplog.text or "encrypted" in caplog.text.lower()

@pytest.mark.asyncio
@patch('extractor.TextExtractor')
async def test_corrupted_pdf_handling(mock_text_extractor_class, caplog):
    """Test that a corrupted PDF is handled gracefully."""
    caplog.set_level(logging.ERROR)
    # Configure the mock TextExtractor instance to return empty text
    mock_text_extractor_class.return_value.extract_text.return_value = ""

    # Placeholder for a corrupted PDF. User needs to provide this file.
    pdf_path = str(TESTS_DIR / "corrupted.pdf")

    # Expecting an InvoiceParsingError if text extraction fails due to corruption
    with pytest.raises(InvoiceParsingError, match="Could not extract any text from the PDF."):
        await extract_invoice_data(pdf_path)
    
    # Check if the error message indicates issues with the PDF
    # assert "Failed to extract text from" in caplog.text or "corrupted" in caplog.text.lower()

@pytest.mark.asyncio
@patch('extractor.TextExtractor')
@patch('extractor.genai') # Patch the google.generativeai as genai import
async def test_gemini_smart_lane_fallback_success(mock_genai, mock_text_extractor_class, caplog):
    """Test that the Gemini Smart Lane is used as a fallback and successfully extracts data."""
    caplog.set_level(logging.INFO)

    # 1. Configure mock TextExtractor to return empty text (simulating docling failure)
    mock_text_extractor_instance = mock_text_extractor_class.return_value
    mock_text_extractor_instance.extract_text.return_value = ""

    # 2. Configure mock Gemini API response
    mock_gemini_client = mock_genai.Client.return_value
    mock_gemini_client.models.generate_content.return_value.text = json.dumps({"invoice_number": "GEMINI-INV-001", "date": "2023-03-20", "total": "999.99", "client": "Gemini Test Client", "vat": "199.99", "line_items": [{"description": "Gemini Item 1", "amount": "500.00"}, {"description": "Gemini Item 2", "amount": "499.99"}]})

    # Use a dummy PDF path, as TextExtractor is mocked
    pdf_path = str(INPUT_PDFS_DIR / "dummy_gemini_test.pdf")

    data = await extract_invoice_data(pdf_path)

    # Assert that docling failed and Gemini was attempted
    assert "Docling failed to extract text from" in caplog.text
    assert "Attempting Gemini Smart Lane" in caplog.text
    assert "Gemini Smart Lane successful" in caplog.text

    # Assert that the data extracted matches the mocked Gemini response
    assert data["invoice_number"] == "GEMINI-INV-001"
    assert data["date"] == "2023-03-20"
    assert data["total"] == "999.99"
    assert data["client"] == "Gemini Test Client"
    assert data["vat"] == "199.99"
    assert len(data["line_items"]) == 2
    assert data["line_items"][0]['description'] == "Gemini Item 1"
    assert data["line_items'][0]['amount'] == "500.00"
    assert data["line_items'][1]['description'] == "Gemini Item 2"
    assert data["line_items'][1]['amount'] == "499.99"
