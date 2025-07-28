import os
import pandas as pd
import logging
from utils import FileSystemUtils

class ReportGenerator:
    """Generates various reports from extracted invoice data."""
    def __init__(self, output_dir: str):
        """Initializes the ReportGenerator with the output directory.

        Args:
            output_dir (str): The directory where reports will be saved.
        """
        self.output_dir = output_dir
        FileSystemUtils.ensure_directory_exists(self.output_dir)

    def generate_report(self, validated_data: dict) -> None:
        """Generates a text report for a single invoice.

        Args:
            validated_data (dict): A dictionary containing the validated data and any validation errors.
        """
        data = validated_data['data']
        errors = validated_data['validation_errors']
        invoice_number = data.get("invoice_number", "UNKNOWN")
        report_filename = os.path.join(self.output_dir, f"invoice_report_{invoice_number}.txt")
        
        with open(report_filename, 'w') as f:
            f.write("--- Invoice Data Report ---\n")
            for key, value in data.items():
                if key == 'line_items':
                    continue
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            if data.get('line_items'):
                f.write("\n--- Line Items ---\n")
                for item in data['line_items']:
                    f.write(f"  - Description: {item['description']}\n")
                    f.write(f"    Amount: {item['amount']}\n")
            
            if errors:
                f.write("\n--- Validation Errors ---\n")
                for error in errors:
                    f.write(f"- {error}\n")

            f.write("---------------------------\n")
        logging.info(f"Generated report for invoice {invoice_number} at {report_filename}")

    def export_to_csv_excel(self, all_validated_data: list) -> None:
        """Exports all extracted invoice data to CSV and Excel files.

        Args:
            all_validated_data (list): A list of dictionaries, where each dictionary contains validated invoice data.
        """
        processed_data = []
        for validated_invoice in all_validated_data:
            invoice_data = validated_invoice['data']
            errors = validated_invoice['validation_errors']
            processed_row = {
                "invoice_number": invoice_data.get('invoice_number', ''),
                "date": invoice_data.get('date', ''),
                "client": invoice_data.get('client', ''),
                "total": invoice_data.get('total', ''),
                "vat": invoice_data.get('vat', ''),
                "currency": invoice_data.get('currency', '')
            }
            processed_data.append(processed_row)

        df = pd.DataFrame(processed_data)
        
        for col in ['total', 'vat']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        csv_path = os.path.join(self.output_dir, 'all_invoices.csv')
        excel_path = os.path.join(self.output_dir, 'all_invoices.xlsx')
        
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        logging.info(f"Exported all data to {csv_path} and {excel_path}")

    def generate_summary_report(self, all_validated_data: list) -> None:
        """Generates a summary report of all extracted invoice data.

        Args:
            all_validated_data (list): A list of dictionaries, where each dictionary contains validated invoice data.
        """
        
        df_data = [d['data'] for d in all_validated_data]
        df = pd.DataFrame(df_data)
        
        for col in ['total', 'vat']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        summary = df.groupby('client').agg(
            total_amount=('total', 'sum'),
            total_vat=('vat', 'sum'),
            invoice_count=('invoice_number', 'count')
        ).reset_index()

        summary_path = os.path.join(self.output_dir, 'summary_report.csv')
        summary.to_csv(summary_path, index=False)
        logging.info(f"Generated summary report at {summary_path}")
