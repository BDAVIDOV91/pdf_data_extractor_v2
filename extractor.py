import logging
import re
import yaml
import json
import os
from google.cloud import vision, storage
from config import settings
from exceptions import InvoiceParsingError
from validator import Validator
from google import genai
from google.genai import types



# --- Configuration for Patterns ---
try:
    with open("patterns.yml", "r") as f:
        PATTERNS = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("patterns.yml not found. The 'Fast Lane' extractor will not work.")
    PATTERNS = {}

# --- Set Google Cloud Credentials ---
# This line tells the Google Cloud library where to find the key file.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

def _extract_text_with_vision_api(pdf_path: str) -> str:
    """
    Extracts text from a PDF using the Google Cloud Vision API.
    This function now handles uploading the PDF to GCS as required by the API
    for asynchronous processing.
    """
    try:
        client = vision.ImageAnnotatorClient()
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(settings.gcs_bucket_name)

        # --- 1. Upload Source PDF to GCS ---
        file_name = os.path.basename(pdf_path)
        gcs_source_uri = f"gs://{settings.gcs_bucket_name}/uploads/{file_name}"
        upload_blob = bucket.blob(f"uploads/{file_name}")

        logging.info(f"Uploading {pdf_path} to {gcs_source_uri}...")
        upload_blob.upload_from_filename(pdf_path)
        logging.info("Upload complete.")

        # --- 2. Call API with GCS Path ---
        gcs_source = vision.GcsSource(uri=gcs_source_uri)
        input_config = vision.InputConfig(
            gcs_source=gcs_source, mime_type="application/pdf"
        )

        # Location to store the results
        gcs_destination_uri = f"gs://{settings.gcs_bucket_name}/results/"
        gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
        output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=1)

        feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

        async_request = vision.AsyncAnnotateFileRequest(
            features=[feature], input_config=input_config, output_config=output_config
        )

        operation = client.async_batch_annotate_files(requests=[async_request])
        logging.info(f"Waiting for Google Vision API operation to complete for {pdf_path}...")
        operation.result(timeout=420)
        logging.info(f"Google Vision API operation completed for {pdf_path}.")

        # --- 3. Retrieve Results ---
        blob_list = list(bucket.list_blobs(prefix="results/"))
        full_text = ""
        result_blobs_to_delete = []

        for blob in blob_list:
            if ".json" in blob.name:
                json_string = blob.download_as_string()
                response = vision.AnnotateFileResponse.from_json(json_string)
                for page_response in response.responses:
                    full_text += page_response.full_text_annotation.text
                result_blobs_to_delete.append(blob)

        logging.info(f"Successfully extracted text for {pdf_path} from GCS.")

        # --- 4. Cleanup ---
        logging.info("Cleaning up GCS files...")
        upload_blob.delete()
        logging.info(f"Deleted uploaded file: {gcs_source_uri}")
        for blob in result_blobs_to_delete:
            blob.delete()
            logging.info(f"Deleted result file: {blob.name}")
        logging.info("Cleanup complete.")

        return full_text

    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path} using Google Vision API: {e}")
        # Ensure cleanup happens even on error if upload_blob was created
        try:
            if 'upload_blob' in locals() and upload_blob.exists():
                upload_blob.delete()
                logging.info(f"Cleaned up orphaned upload file: {gcs_source_uri}")
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {cleanup_error}")
        return ""







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
    Orchestrates invoice data extraction using a dual-pathway logic.
    It first attempts text extraction for the 'Fast Lane' (regex patterns).
    If that fails or yields insufficient data, it falls back to the 'Smart Lane' (Gemini API).
    """
    text = _extract_text_with_vision_api(pdf_path)

    # --- Pathway 1: The "Fast Lane" ---
    if text:
        logging.info(f"Attempting 'Fast Lane' extraction for {pdf_path}...")
        fast_lane_data = _extract_data_with_patterns(text)

        if _is_data_sufficient(fast_lane_data):
            logging.info(f"'Fast Lane' extraction successful for {pdf_path}.")
            # Ensure all keys are present, even if None
            final_data = {
                "invoice_number": fast_lane_data.get("invoice_number"),
                "date": Validator._normalize_date(fast_lane_data.get("date", "")),
                "total": fast_lane_data.get("total"),
                "client": fast_lane_data.get("client"),
                "vat": fast_lane_data.get("vat"),
                "line_items": [], # Placeholder, line item extraction is complex for regex
            }
            return final_data

    # --- Pathway 2: The "Smart Lane" (Fallback) ---
    logging.info(f"'Fast Lane' failed or text extraction yielded no results. Switching to 'Smart Lane' for {pdf_path}.")
    if settings.smart_lane_enabled:
        gemini_data = _extract_data_with_gemini(pdf_path)
        if gemini_data:
            logging.info(f"Gemini 'Smart Lane' successful for {pdf_path}.")
            # The validation function will handle normalization and structure
            return gemini_data

    # If all pathways fail
    raise InvoiceParsingError(f"Could not extract sufficient data from {pdf_path} using any available method.")
