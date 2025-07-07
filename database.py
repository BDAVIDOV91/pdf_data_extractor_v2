import sqlite3
import logging

class DatabaseManager:
    """Handles all database operations for storing invoice data."""
    def __init__(self, db_path: str):
        """Initializes the DatabaseManager and connects to the database.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logging.info(f"Successfully connected to database at {self.db_path}")
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}")
            raise

    def create_tables(self) -> None:
        """Creates the necessary tables if they don't already exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_number TEXT UNIQUE NOT NULL,
                    date TEXT,
                    client TEXT,
                    total REAL,
                    vat REAL,
                    currency TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS line_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id INTEGER,
                    description TEXT,
                    amount REAL,
                    FOREIGN KEY (invoice_id) REFERENCES invoices (id)
                )
            """)
            self.conn.commit()
            logging.info("Database tables created or already exist.")
        except sqlite3.Error as e:
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
            cursor.execute("""
                INSERT OR REPLACE INTO invoices (invoice_number, date, client, total, vat, currency)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.get('invoice_number'),
                data.get('date'),
                data.get('client'),
                data.get('total'),
                data.get('vat'),
                data.get('currency')
            ))
            invoice_id = cursor.lastrowid
            
            if invoice_id and data.get('line_items'):
                for item in data['line_items']:
                    cursor.execute("""
                        INSERT INTO line_items (invoice_id, description, amount)
                        VALUES (?, ?, ?)
                    """, (invoice_id, item.get('description'), item.get('amount')))
            
            self.conn.commit()
            logging.info(f"Successfully inserted invoice {data.get('invoice_number')} into the database.")
        except sqlite3.Error as e:
            logging.error(f"Failed to insert invoice {data.get('invoice_number')}: {e}")
            self.conn.rollback()

    def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
