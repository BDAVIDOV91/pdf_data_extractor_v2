import os
import cv2
import numpy as np
from PIL import Image

class FileSystemUtils:
    @staticmethod
    def ensure_directory_exists(directory: str):
        """Ensures that the specified directory exists. If it does not, it creates it.

        Args:
            directory (str): The path to the directory to ensure its existence.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def get_file_name_without_extension(file_path: str) -> str:
        """Extracts the file name without its extension from a given file path.

        Args:
            file_path (str): The full path to the file.

        Returns:
            str: The file name without its extension.
        """
        base_name = os.path.basename(file_path)
        file_name_without_extension, _ = os.path.splitext(base_name)
        return file_name_without_extension

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Extracts the file extension from a given file path. 

        Args:
            file_path (str): The full path to the file.

        Returns:
            str: The file extension (e.g., ".pdf", ".txt").
        """
        _, file_extension = os.path.splitext(file_path)
        return file_extension

    @staticmethod
    def is_pdf(file_path: str) -> bool:
        """Checks if a given file path points to a PDF file.

        Args:
            file_path (str): The path to the file.

        Returns:
            bool: True if the file is a PDF, False otherwise.
        """
        return FileSystemUtils.get_file_extension(file_path).lower() == ".pdf"

    @staticmethod
    def is_image(file_path: str) -> bool:
        """Checks if a given file path points to an image file.

        Args:
            file_path (str): The full path to the file.

        Returns:
            bool: True if the file is an image, False otherwise.
        """
        return FileSystemUtils.get_file_extension(file_path).lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Checks if a given file path points to a common text file.

        Args:
            file_path (str): The path to the file.

        Returns:
            bool: True if the file is a text file, False otherwise.
        """
        return FileSystemUtils.get_file_extension(file_path).lower() in [".txt", ".log", ".csv", ".json", ".xml", ".yaml", ".yml"]

    @staticmethod
    def get_all_files_in_directory(directory: str, extension: str = None) -> list:
        """Retrieves a list of all files in a given directory, optionally filtered by extension.

        Args:
            directory (str): The path to the directory.
            extension (str, optional): The file extension to filter by (e.g., ".pdf"). Defaults to None.

        Returns:
            list: A list of absolute paths to the files.
        """
        files = []
        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                if extension is None or FileSystemUtils.get_file_extension(full_path).lower() == extension.lower():
                    files.append(full_path)
        return files

    @staticmethod
    def get_relative_path(base_path: str, target_path: str) -> str:
        """Calculates the relative path from a base path to a target path.

        Args:
            base_path (str): The base directory path.
            target_path (str): The target file or directory path.

        Returns:
            str: The relative path.
        """
        return os.path.relpath(target_path, base_path)

    @staticmethod
    def get_absolute_path(relative_path: str, base_path: str) -> str:
        """Calculates the absolute path from a relative path and a base path.

        Args:
            relative_path (str): The relative path.
            base_path (str): The base directory path.

        Returns:
            str: The absolute path.
        """
        return os.path.join(base_path, relative_path)

    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """Applies image pre-processing steps to enhance OCR accuracy.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The processed image.
        """
        img_cv = np.array(image.convert('L')) # Convert to grayscale
        
        # Deskewing
        coords = np.column_stack(np.where(img_cv > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Apply adaptive thresholding for better handling of varying lighting
        thresh = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Median blur for noise reduction
        denoised = cv2.medianBlur(thresh, 3)
        
        processed_image = Image.fromarray(denoised)
        
        return processed_image