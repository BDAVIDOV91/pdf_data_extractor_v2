import os

class FileSystemUtils:
    @staticmethod
    def ensure_directory_exists(directory: str):
        """Ensures that the specified directory exists. If it does not, it creates it.

        Args:
            directory (str): The path to the directory to ensure its existence.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)