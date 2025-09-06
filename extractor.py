import logging
import re
import yaml
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.easyocr_model import EasyOcrOptions
from config import settings
from exceptions import InvoiceParsingError
from validator import Validator
from google import genai
from google.genai import types

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration for Patterns ---
try:
    with open("patterns.yml", "r") as f:
        PATTERNS = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("patterns.yml not found. The 'Fast Lane' extractor will not work.")
    PATTERNS = {}


class TextExtractor:
    """Handles text extraction from PDF files, using docling."""

    def __init__(self):
        """Initializes the TextExtractor."""
        # Configure EasyOCR options
        easyocr_options = EasyOcrOptions(
            lang=["en"],  # EasyOCR uses 'en' for English
        )

        # Configure PDF pipeline options to enable OCR and use EasyOCR
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,  # Enable OCR
            ocr_options=easyocr_options,  # Specify the EasyOCR options
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_text(self, pdf_path: str) -> str:
        """Extracts text from a PDF using docling."""
        try:
            result = self.converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logging.error(f"Failed to extract text from {pdf_path} using docling: {e}")
            return ""


class QuestionAnswering:
    """Handles question answering using a RAG architecture (The "Smart Lane")."""

    def __init__(self):
        """Initializes the QuestionAnswering class."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            is_separator_regex=False
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )
        self.llm = OllamaLLM(model="qwen:1.8b")

    def create_vector_store(self, texts: list[str]) -> FAISS:
        """Create and initialize FAISS vector store using text embeddings"""
        vector_store = FAISS.from_texts(texts, self.embeddings)
        return vector_store

    def get_qa_chain(self, vector_store):
        """Create question-answering chain using LLM and vector store"""
        prompt_template = """
            Use the following pieces of context to answer the question at the end.

            Check context very carefully and reference and try to make sense of that before responding.
            If you don't know the answer, just say you don't know.
            Don't try to make up an answer.
            Answer must be to the point.
            Think step-by-step.

            Context: {context}

            Question: {question}

            Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )
        return qa_chain

    def answer_question(self, text: str, question: str) -> str:
        """Answers a question about the given text."""
        text_chunks = self.text_splitter.split_text(text)
        vector_store = self.create_vector_store(text_chunks)
        qa_chain = self.get_qa_chain(vector_store)
        response = qa_chain.invoke({"query": question})
        return response['result']

# --- Parsing functions for LLM output ---

def _extract_data_with_gemini(pdf_path: str) -> dict | None:
    """
    Extracts invoice data using Gemini API for complex/image-based PDFs.
    This serves as an alternative Smart Lane entry point.
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_bytes = file.read()

        # Prompt for Gemini to extract structured invoice data
        prompt = """Extract the following fields from the document:
        Invoice Number, Date (YYYY-MM-DD), Total Amount, Client Name, VAT Amount.
        If line items are present, extract them with Description and Amount.
        Provide the output as a JSON object.
        Example:
        {
            "invoice_number": "INV-2023-001",
            "date": "2023-01-15",
            "total": "150.00",
            "client": "ABC Corp",
            "vat": "20.00",
            "line_items": [
                {"description": "Item 1", "amount": "100.00"},
                {"description": "Item 2", "amount": "50.00"}
            ]
        }
        If a field is not found, use "N/A" for string fields and "0.00" for numeric fields.
        """
        
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash", # Using a fast model for extraction
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )
        
        # Attempt to parse the response as JSON
        try:
            # Attempt to parse the response as JSON
            json_string = response.text.strip()
            if json_string.startswith("```json") and json_string.endswith("```"):
                json_string = json_string[len("```json"):-len("```")].strip()
            extracted_json = json.loads(json_string)
            logging.info(f"Gemini extraction successful for {pdf_path}.")
            return extracted_json
        except json.JSONDecodeError:
            logging.error(f"Gemini response is not valid JSON for {pdf_path}: {response.text}")
            return None

    except Exception as e:
        logging.error(f"Error during Gemini extraction for {pdf_path}: {e}")
        return None

def _parse_invoice_number(text: str) -> str:
    """Extracts and cleans the invoice number from the LLM's response."""
    match = re.search(r'([A-Z0-9-]+)', text)
    return match.group(0).strip() if match else "N/A"

def _parse_total(text: str) -> str:
    """Extracts and cleans the total amount from the LLM's response."""
    match = re.search(r'[\d,.]+', text)
    return match.group(0).replace(',', '').strip() if match else "N/A"

def _parse_client(text: str) -> str:
    """Extracts and cleans the client name from the LLM's response."""
    return text.strip() if text else "N/A"

def _parse_vat(text: str) -> str:
    """Extracts and cleans the VAT amount from the LLM's response."""
    match = re.search(r'[\d,.]+', text)
    return match.group(0).replace(',', '').strip() if match else "N/A"

def _parse_line_items_from_text(text: str) -> list[dict]:
    """Parses line items from a simple text response, robust to inconsistent LLM output."""
    line_items = []
    lines = text.strip().split('\n')

    if not lines:
        return []

    # Attempt to clean conversational text
    cleaned_lines = []
    for line in lines:
        # Simple heuristic to remove common conversational intros/outros
        if any(phrase in line.lower() for phrase in ["here are the line items:", "i hope this helps", "description;amount"]):
            continue
        cleaned_lines.append(line)
    
    if not cleaned_lines:
        return []

    # Try to find a header line or infer structure
    possible_delimiters = [';', ',', '\t'] # Semicolon, comma, tab
    
    headers = []
    data_lines = []

    # Heuristic to find header: look for common keywords in the first few lines
    potential_header_keywords = ["description", "item", "amount", "price", "quantity", "total"]
    
    for i, line in enumerate(cleaned_lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in potential_header_keywords):
            # This might be our header line
            for delim in possible_delimiters:
                parts = [p.strip() for p in line.split(delim) if p.strip()]
                if len(parts) > 1 and any(p.lower() in potential_header_keywords for p in parts):
                    headers = [h.lower().replace(" ", "_") for h in parts]
                    data_lines = cleaned_lines[i+1:]
                    break
            if headers:
                break
    
    if not headers: # Fallback: if no clear header, assume a simple description;amount structure
        logging.warning("No clear header found for line items. Assuming 'description;amount' structure.")
        headers = ["description", "amount"]
        data_lines = cleaned_lines

    if not headers: # Still no headers, something is wrong
        logging.warning("Could not determine headers for line items.")
        return []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        
        values = []
        for delim in possible_delimiters:
            temp_values = [v.strip() for v in line.split(delim) if v.strip()]
            if len(temp_values) >= len(headers): # Found a delimiter that gives enough values
                values = temp_values
                break
        
        if not values or len(values) < len(headers):
            logging.warning(f"Skipping malformed line item row: {line}")
            continue
        
        item_data = {}
        for i, header in enumerate(headers):
            if i < len(values):
                item_data[header] = values[i]
        line_items.append(item_data)
    
    return line_items

# --- "Fast Lane" and Orchestration Logic ---

def _extract_data_with_patterns(text: str) -> dict | None:
    """
    Attempts to extract invoice data using predefined regex patterns from patterns.yml.
    This is the "Fast Lane".
    """
    if not PATTERNS:
        return None

    selected_pattern = None
    for name, pattern_set in PATTERNS.items():
        if any(re.search(keyword, text, re.IGNORECASE) for keyword in pattern_set.get("keywords", [])):
            selected_pattern = pattern_set
            logging.info(f"Matched pattern set: '{name}'")
            break

    if not selected_pattern:
        logging.info("No specific pattern matched. Falling back to generic document pattern.")
        selected_pattern = PATTERNS.get("document", {})

    data = {}
    for key, pattern in selected_pattern.items():
        if key in ["invoice_number", "date", "total", "client", "vat"]:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            data[key] = match.group(1).strip() if match else None
            if key in ["invoice_number", "date", "total"]:
                logging.info(f"Regex search for '{key}': {'Success' if match else 'Failure'}")

    return data

def _is_data_sufficient(data: dict) -> bool:
    logging.info(f"Data from Fast Lane: {data}")
    """
    Checks if the extracted data from the "Fast Lane" is good enough.
    """
    if not data:
        return False
    # Essential fields that must be present for the fast lane to be considered successful
    essential_fields = ["invoice_number", "date", "total"]
    return all(data.get(field) for field in essential_fields)

async def extract_invoice_data(pdf_path: str) -> dict:
    """
    Orchestrates invoice data extraction using the dual-pathway logic.
    It first tries the "Fast Lane" (regex patterns) and falls back to the
    "Smart Lane" (AI/LLM) if the initial results are insufficient.
    """
    validator_instance = Validator() # Instantiate Validator once
    text_extractor = TextExtractor()
    text = text_extractor.extract_text(pdf_path)

    if not text:
        logging.info(f"Docling failed to extract text from {pdf_path}. Attempting Gemini Smart Lane...")
        gemini_data = _extract_data_with_gemini(pdf_path)
        if gemini_data:
            logging.info(f"Gemini Smart Lane successful for {pdf_path}.")
            # Validate and normalize Gemini's output to match expected structure
            validated_gemini_data = validator_instance.validate_and_normalize_data({
                "invoice_number": gemini_data.get("invoice_number"),
                "date": gemini_data.get("date"),
                "total": gemini_data.get("total"),
                "client": gemini_data.get("client"),
                "vat": gemini_data.get("vat"),
                "line_items": gemini_data.get("line_items", [])
            })
            return validated_gemini_data['data'] # Return only the data part
        else:
            raise InvoiceParsingError("Could not extract any text from the PDF using Docling or Gemini.")

    # --- Pathway 1: The "Fast Lane" ---
    logging.info(f"Attempting 'Fast Lane' extraction for {pdf_path}...")
    fast_lane_data = _extract_data_with_patterns(text)

    if _is_data_sufficient(fast_lane_data):
        logging.info(f"'Fast Lane' extraction successful for {pdf_path}.")
        # Ensure all keys are present, even if None
        final_data = {
            "invoice_number": fast_lane_data.get("invoice_number"),
            "date": validator_instance._normalize_date(fast_lane_data.get("date", "")),
            "total": fast_lane_data.get("total"),
            "client": fast_lane_data.get("client"),
            "vat": fast_lane_data.get("vat"),
            "line_items": [], # Placeholder, line item extraction is complex for regex
        }
        return final_data

    # --- Pathway 2: The "Smart Lane" (Fallback) ---
    logging.info(f"'Fast Lane' failed or data insufficient. Switching to 'Smart Lane' for {pdf_path}.")
    qa = QuestionAnswering()

    # Extract Invoice Number
    invoice_number_raw = qa.answer_question(text, "What is the invoice number? Respond with only the number.")
    invoice_number = _parse_invoice_number(invoice_number_raw)

    # Extract Date
    date_raw = qa.answer_question(text, "What is the invoice date? Respond with only the date in YYYY-MM-DD format.")
    date = validator_instance._normalize_date(date_raw)

    # Extract Total Amount
    total_raw = qa.answer_question(text, "What is the total amount of the invoice? Respond with only the number.")
    total = _parse_total(total_raw)

    # Extract Client Name
    client_raw = qa.answer_question(text, "What is the client name on the invoice? Respond with only the client name.")
    client = _parse_client(client_raw)

    # Extract VAT
    vat_raw = qa.answer_question(text, "What is the VAT amount on the invoice? Respond with only the number.")
    vat = _parse_vat(vat_raw)
    
    # Extract Line Items (Complex task, remains in Smart Lane)
    question_line_items = "For each line item, provide the description, quantity, unit price, and total price, separated by semicolons. Each line item on a new line. Do NOT include any other text, explanations, or conversational elements. Only provide the semicolon-separated data."
    line_items_raw = qa.answer_question(text, question_line_items)
    line_items = _parse_line_items_from_text(line_items_raw)


    smart_lane_data = {
        "invoice_number": invoice_number,
        "date": date,
        "total": total,
        "client": client,
        "line_items": line_items,
        "vat": vat,
    }

    return smart_lane_data