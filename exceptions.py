class PDFExtractionError(Exception):
    """Base exception for PDF extraction errors."""
    pass

class TextExtractionError(PDFExtractionError):
    """Raised when text cannot be extracted from a PDF."""
    pass

class OCRProcessingError(TextExtractionError):
    """Raised when an error occurs during OCR processing."""
    pass

class InvoiceParsingError(PDFExtractionError):
    """Raised when an error occurs during invoice parsing."""
    pass

class UnsupportedInvoiceFormatError(InvoiceParsingError):
    """Raised when the invoice format is not supported."""
    pass
