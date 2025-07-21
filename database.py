import psycopg2
from psycopg2 import Error
import logging

class DatabaseManager:
    """Handles all database operations for storing invoice data."""
    def __init__(self, db_url: str):
        """Initializes the DatabaseManager and connects to the database.

        Args:
            db_url (str): The URL for the PostgreSQL database.
        """
        self.db_url = db_url
        self.conn = None
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = True  # Auto-commit for DDL operations
            logging.info(f"Successfully connected to database at {self.db_url}")
        except Error as e:
            logging.error(f"Database connection failed: {e}")
            raise

    def create_tables(self) -> None:
        """Creates the necessary tables if they don't already exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id SERIAL PRIMARY KEY,
                    invoice_number TEXT UNIQUE NOT NULL,
                    date TEXT,
                    client TEXT,
                    total REAL,
                    vat REAL,
                    currency TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS line_items (
                    id SERIAL PRIMARY KEY,
                    invoice_id INTEGER REFERENCES invoices(id) ON DELETE CASCADE,
                    description TEXT,
                    amount REAL
                );
            """)
            logging.info("Database tables created or already exist.")
        except Error as e:
            logging.error(f"Table creation failed: {e}")

    def insert_invoice(self, validated_data: dict) -> None:
        """Inserts a validated invoice and its line items into the database.

        Args:
            validated_data (dict): A dictionary containing validated data and errors.
        """
        data = validated_data['data']
        if validated_data['validation_errors']:
            logging.warning(f"Skipping database insertion for invoice {data.get('invoice_number')} due to validation errors.")
            return

        try:
            cursor = self.conn.cursor()
            # Check if invoice_number already exists
            cursor.execute("SELECT id FROM invoices WHERE invoice_number = %s", (data.get('invoice_number'),))
            existing_invoice = cursor.fetchone()

            if existing_invoice:
                invoice_id = existing_invoice[0]
                # Update existing invoice
                cursor.execute("""
                    UPDATE invoices
                    SET date = %s, client = %s, total = %s, vat = %s, currency = %s
                    WHERE id = %s
                """, (
                    data.get('date'),
                    data.get('client'),
                    data.get('total'),
                    data.get('vat'),
                    data.get('currency'),
                    invoice_id
                ))
                # Delete old line items and insert new ones
                cursor.execute("DELETE FROM line_items WHERE invoice_id = %s", (invoice_id,))
            else:
                # Insert new invoice
                cursor.execute("""
                    INSERT INTO invoices (invoice_number, date, client, total, vat, currency)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
                """, (
                    data.get('invoice_number'),
                    data.get('date'),
                    data.get('client'),
                    data.get('total'),
                    data.get('vat'),
                    data.get('currency')
                ))
                invoice_id = cursor.fetchone()[0]
            
            if invoice_id and data.get('line_items'):
                for item in data['line_items']:
                    cursor.execute("""
                        INSERT INTO line_items (invoice_id, description, amount)
                        VALUES (%s, %s, %s)
                    """, (invoice_id, item.get('description'), item.get('amount')))
            
            self.conn.commit()
            logging.info(f"Successfully inserted/updated invoice {data.get('invoice_number')} into the database.")
        except Error as e:
            logging.error(f"Failed to insert/update invoice {data.get('invoice_number')}: {e}")
            self.conn.rollback()

    def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
