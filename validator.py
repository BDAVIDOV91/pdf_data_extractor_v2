import logging
from datetime import datetime

class Validator:
    """Handles validation and normalization of extracted invoice data."""

    def validate_and_normalize_data(self, data: dict) -> dict:
        """Runs a series of validation and normalization checks on the data.

        Args:
            data (dict): The raw extracted invoice data.

        Returns:
            dict: A dictionary containing the normalized data and a list of validation errors.
        """
        errors = []
        
        # Determine required fields based on document type
        if data.get("doc_type") == "receipt":
            required_fields = ["date", "total"]
        else:
            required_fields = ["invoice_number", "date", "total"]

        for field in required_fields:
            if data.get(field) is None:
                errors.append(f"Missing required field: {field}")

        # Validate and normalize date
        if "date" in data and data["date"] != "Not found":
            normalized_date = self._normalize_date(data["date"])
            if normalized_date:
                data["date"] = normalized_date
            else:
                errors.append(f"Invalid date format: {data['date']}")

        # (Future) Validate amounts
        # (Future) Validate line item consistency

        validated_data = {
            "data": data,
            "validation_errors": errors
        }
        
        if errors:
            logging.warning(f"Validation failed for invoice {data.get('invoice_number', '')}: {errors}")
        else:
            logging.info(f"Validation successful for invoice {data.get('invoice_number', '')}")
            
        return validated_data

    def _normalize_date(self, date_str: str) -> str | None:
        """Normalizes a date string from various formats to YYYY-MM-DD.

        Args:
            date_str (str): The date string to normalize.

        Returns:
            str | None: The normalized date string or None if parsing fails.
        """
        if not isinstance(date_str, str):
            return None

        date_str = date_str.strip()
        formats_to_try = [
            "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y",
            "%Y.%m.%d", "%Y/%m/%d", "%Y-%m-%d",
            "%m/%d/%Y",
            "%d %b %Y", "%d %B %Y",
        ]

        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        logging.warning(f"Could not parse date: {date_str}")
        return None
