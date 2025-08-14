import base64
import logging
import os
import re
from multiprocessing import Process, Queue
from queue import Empty
from io import BytesIO

import httpx
import PyPDF2
import yaml
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from docling.document_converter import DocumentConverter
import ollama
from parsemypdf.utils.pdf_to_image import PDFToJPGConverter
import shutil


from config import settings
from exceptions import (
    InvoiceParsingError, OCRProcessingError,
    TextExtractionError, UnsupportedInvoiceFormatError
)
from utils import FileSystemUtils, pymupdf_extract


class TextExtractor:
    """Handles text extraction from PDF files, using a tiered approach."""

    def __init__(self, enable_ocr: bool = False):
        """Initializes the TextExtractor.

        Args:
            enable_ocr (bool): Whether to enable OCR for scanned PDFs.
        """
        self.enable_ocr = enable_ocr
        self.ocr_preprocessing_enabled = settings.OCR_PREPROCESSING_ENABLED
        self.jigsaw_api_key = settings.jigsaw_api_key

    def _deskew_image(self, image: Image.Image) -> Image.Image:
        """Deskews an image using OpenCV."""
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)

    def _binarize_image(self, image: Image.Image) -> Image.Image:
        """Binarizes an image using OpenCV (Otsu's method)."""
        img_np = np.array(image.convert('L'))  # Convert to grayscale
        _, binarized = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binarized)

    def _remove_noise(self, image: Image.Image) -> Image.Image:
        """Removes noise from an image using OpenCV (fastNlMeansDenoising)."""
        img_np = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
        return Image.fromarray(denoised)


    async def _extract_text_with_llama_vision(self, pdf_path: str) -> str:
        converter = PDFToJPGConverter()
        output_path = "converted_images/temp"
        os.makedirs(output_path, exist_ok=True)  # Ensure directory exists
        converted_files = converter.convert_pdf(pdf_path, output_path)

        final_response = ""
        for original_file_path in converted_files:
            try:
                image = Image.open(original_file_path)
                
                # Apply pre-processing if enabled
                if self.ocr_preprocessing_enabled:
                    logging.info(f"Applying OCR pre-processing to {original_file_path}")
                    image = self._deskew_image(image)
                    image = self._binarize_image(image)
                    image = self._remove_noise(image)
                
                # Convert processed image back to base64
                with BytesIO() as buffer:
                    image.save(buffer, format="PNG")
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

                response = ollama.chat(
                    model='x/llama3.2-vision:11b',
                    messages=[{
                        'role': 'user',
                        'content': '''You are an expert at extracting and structuring content from image. 
                                        Please extract all the text content from the provided image, maintaining the 
                                        structure and formatting of each element.
                                        Format tables properly in markdown format. Preserve all numerical data and 
                                        relationships between elements as given in the images''',
                        'images': [base64_image]
                    }]
                )
                final_response += response['message']['content'] + "\n"
            except Exception as e:
                logging.error(f"Error processing image {original_file_path} with Llama Vision: {e}")
            finally:
                if os.path.exists(original_file_path):
                    os.remove(original_file_path)  # Clean up temporary image file
        if os.path.exists(output_path):
            shutil.rmtree(output_path)  # Clean up temporary directory
        return final_response


    async def extract_text(self, pdf_path: str, use_ocr: bool) -> str:
        """Extracts text from a PDF, deciding whether to use OCR based on the use_ocr flag."""
        # Tier 1: Try local PyMuPDF (for non-OCR path)
        if not use_ocr:
            try:
                text = pymupdf_extract(pdf_path)
                if text and text.strip():
                    logging.info("Extracted text using local PyMuPDF.")
                    return text
            except Exception as e:
                logging.warning(f"Local PyMuPDF failed: {e}. Consider using OCR path.")

        # Tier 2: Full OCR pipeline (for OCR path)
        if use_ocr:
            logging.info("Attempting OCR extraction pipeline.")
            try:
                # Start with Docling OCR
                converter = DocumentConverter()
                result = converter.convert(pdf_path)
                docling_text = result.document.export_to_markdown()
                if docling_text and len(docling_text.strip()) > 100:
                    logging.info("Extracted text using Docling OCR.")
                    return docling_text
                else:
                    logging.warning("Docling OCR produced insufficient text. Falling back to Llama Vision OCR.")
                    raise OCRProcessingError("Docling OCR failed to extract sufficient text.")
            except Exception as e:
                logging.error(f"Error during Docling OCR processing: {e}")
                logging.warning(f"Docling OCR failed. Falling back to Llama Vision OCR.")
                try:
                    llama_text = await self._extract_text_with_llama_vision(pdf_path)
                    if llama_text and llama_text.strip():
                        logging.info("Extracted text using Llama Vision OCR.")
                        return llama_text
                    else:
                        raise OCRProcessingError("Llama Vision OCR failed to extract text.")
                except Exception as llama_e:
                    logging.error(f"Error during Llama Vision OCR processing: {llama_e}")
                    raise OCRProcessingError(f"Llama Vision OCR processing failed: {llama_e}")
        
        raise TextExtractionError(f"Could not extract text from {pdf_path} using any method.")


class InvoiceParser:
    """Parses invoice data from extracted text using predefined regex patterns or LLMs."""

    def __init__(self):
        """Initializes the InvoiceParser with patterns loaded from patterns.yml."""
        with open("patterns.yml", "r") as f:
            self.patterns = yaml.safe_load(f)

    def normalize_amount(self, amount_str: str) -> float | None:
        """Normalizes an amount string to a float."""
        if not isinstance(amount_str, str):
            return None
        cleaned_amount = amount_str.replace(" ", "")
        last_comma = cleaned_amount.rfind(",")
        last_dot = cleaned_amount.rfind(".")
        if last_comma > last_dot:
            cleaned_amount = cleaned_amount.replace(".", "").replace(",", ".")
        else:
            cleaned_amount = cleaned_amount.replace(",", "")
        try:
            return float(cleaned_amount)
        except ValueError as e:
            logging.warning(f"Normalization failed for amount string '{amount_str}': {e}")
            return None

    def get_document_type(self, text: str) -> str:
        """Determines the document type by scoring pattern matches."""
        best_match = "document"
        max_score = 0
        
        prioritized_doc_types = [
            "vukov_development_services", "etkyusi_eood", "atqc_ltd", 
            "ikea_receipt", "replit", "patent_and_trademark_institute", "receipt"
        ]

        for doc_type in prioritized_doc_types:
            if doc_type not in self.patterns:
                continue
            
            patterns = self.patterns[doc_type]
            current_score = 0

            # Score based on keyword matches (high weight)
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if re.search(keyword, text, re.IGNORECASE):
                        current_score += 10

            # Score based on specific field matches (lower weight)
            for field, pattern in patterns.items():
                if field not in ["keywords", "document_type", "line_items", "line_items_block"]:
                    if re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.UNICODE):
                        current_score += 1
            
            logging.debug(f"Document type: {doc_type}, Score: {current_score}")

            if current_score > max_score:
                max_score = current_score
                best_match = doc_type

        logging.info(f"Identified document type as: {best_match} with score {max_score}")
        return best_match

    def parse_line_items(self, text: str, doc_type: str) -> list:
        """Parses line items from the invoice text."""
        items = []
        layout_patterns = self.patterns.get(doc_type, {})
        line_items_block_pattern = layout_patterns.get("line_items_block")
        line_items_pattern = layout_patterns.get("line_items")
        if not line_items_pattern:
            return items
        text_block = text
        if line_items_block_pattern:
            block_match = re.search(line_items_block_pattern, text, re.IGNORECASE | re.DOTALL | re.UNICODE)
            if block_match:
                text_block = block_match.group(0)
        for match in re.finditer(line_items_pattern, text_block, re.IGNORECASE | re.DOTALL | re.UNICODE):
            item_data = match.groupdict()
            description = item_data.get("description", "").strip()
            amount, quantity, unit_price = None, None, None
            numbers_str = item_data.get("numbers")
            if numbers_str:
                numeric_values = [self.normalize_amount(s) for s in re.findall(r'[\d.,]+', numbers_str)]
                numeric_values = [val for val in numeric_values if val is not None]
                if len(numeric_values) >= 1:
                    amount = numeric_values[-1]
                if len(numeric_values) >= 2:
                    unit_price = numeric_values[-2]
                if len(numeric_values) >= 3:
                    quantity = numeric_values[0]
            if description or amount is not None:
                items.append({"description": description, "amount": amount, "quantity": quantity, "unit_price": unit_price})
        return items

    def parse_invoice_data_with_patterns(self, text: str) -> dict:
        """Parses invoice data from text using regex patterns (Fast Lane)."""
        data = {"invoice_number": None, "date": None, "client": None, "line_items": [], "total": None, "vat": None, "currency": None, "transaction_id": None}
        if text is None:
            raise InvoiceParsingError("Cannot parse invoice data: text is None.")
        doc_type = self.get_document_type(text)
        layout_patterns = self.patterns.get(doc_type, self.patterns.get("document"))
        if not layout_patterns:
            raise InvoiceParsingError(f"No patterns found for document type: {doc_type}")
        for field in ["invoice_number", "date", "client", "total", "vat"]:
            if field in layout_patterns:
                match = re.search(layout_patterns[field], text, re.IGNORECASE | re.UNICODE | (re.DOTALL if field == "client" else 0))
                if match:
                    extracted_value = match.group(1).strip()
                    if field in ["total", "vat"]:
                        data[field] = self.normalize_amount(extracted_value) or 0.0
                    else:
                        data[field] = extracted_value.replace("\n", " ") if field == "client" else extracted_value.replace(" ", "")
        if doc_type == "receipt" and "transaction_id" in layout_patterns:
            match = re.search(layout_patterns["transaction_id"], text, re.IGNORECASE)
            if match:
                data["transaction_id"] = match.group(1).strip()
        if doc_type == "patent_and_trademark_institute" and "total" in layout_patterns:
            all_totals = re.findall(layout_patterns["total"], text, re.IGNORECASE | re.UNICODE)
            if all_totals:
                total_val = all_totals[-1]
                if isinstance(total_val, tuple):
                    total_val = next((item for item in total_val if item), None)
                data["total"] = self.normalize_amount(total_val.strip())
        if data["total"] is not None:
            if re.search(r"BGN|лв", text, re.IGNORECASE):
                data["currency"] = "BGN"
            elif re.search(r"EUR|€", text, re.IGNORECASE):
                data["currency"] = "EUR"
            else:
                data["currency"] = "EUR"
        data["line_items"] = self.parse_line_items(text, doc_type)
        logging.info(f"Fast Lane extracted data: {data}") # DEBUGGING
        return data

    async def parse_invoice_data_with_llm(self, text: str) -> dict:
        """Parses invoice data from text using an LLM (Smart Lane)."""
        logging.info("Parsing invoice data using the Smart Lane (LLM)...")
        # Placeholder for LLM-based parsing logic
        # For now, returns a default structure
        data = {"invoice_number": "LLM_PLACEHOLDER", "date": None, "client": None, "line_items": [], "total": None, "vat": None, "currency": None, "transaction_id": None}
        # Here you would structure a prompt, call the LLM, and parse its response.
        logging.warning("Smart Lane (LLM) parsing is not yet implemented.")
        return data

async def extract_invoice_data(pdf_path: str, enable_ocr: bool, invoice_parser: InvoiceParser = None) -> dict:
    """Orchestrates invoice data extraction using the dual-pathway logic."""
    if invoice_parser is None:
        invoice_parser = InvoiceParser()
    
    text_extractor = TextExtractor(enable_ocr=enable_ocr)

    # 1. Attempt Fast Lane extraction first
    try:
        initial_text = await text_extractor.extract_text(pdf_path, use_ocr=False)
        if initial_text and initial_text.strip():
            logging.info("Attempting Fast Lane (pattern-based) parsing.")
            data = invoice_parser.parse_invoice_data_with_patterns(initial_text)
            # 2. Check if Fast Lane result is sufficient
            if data.get("invoice_number") and data.get("total") is not None:
                logging.info("Fast Lane parsing successful.")
                return data
            else:
                logging.warning("Fast Lane parsing yielded incomplete data. Falling back to Smart Lane.")
        else:
            logging.info("Initial text extraction yielded no content. Proceeding to Smart Lane.")

    except Exception as e:
        logging.warning(f"Fast Lane failed: {e}. Falling back to Smart Lane.")

    # 3. Fallback to Smart Lane
    logging.info("Executing Smart Lane (OCR + LLM).")
    ocr_text = await text_extractor.extract_text(pdf_path, use_ocr=True)
    return await invoice_parser.parse_invoice_data_with_llm(ocr_text)
