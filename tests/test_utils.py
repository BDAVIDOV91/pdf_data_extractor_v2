import unittest
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FileSystemUtils

class TestFileSystemUtils(unittest.TestCase):

    def setUp(self):
        self.test_dir = "./test_dir"
        self.pdf_dir = os.path.join(self.test_dir, "pdfs")
        self.image_dir = os.path.join(self.test_dir, "images")
        FileSystemUtils.ensure_directory_exists(self.pdf_dir)
        FileSystemUtils.ensure_directory_exists(self.image_dir)
        
        # Create dummy files for testing
        with open(os.path.join(self.pdf_dir, "test1.pdf"), "w") as f:
            f.write("dummy pdf")
        with open(os.path.join(self.pdf_dir, "test2.PDF"), "w") as f:
            f.write("dummy pdf")
        with open(os.path.join(self.pdf_dir, "not_a_pdf.txt"), "w") as f:
            f.write("dummy text")
        with open(os.path.join(self.image_dir, "image1.png"), "w") as f:
            f.write("dummy image")
        with open(os.path.join(self.image_dir, "image2.JPG"), "w") as f:
            f.write("dummy image")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ensure_directory_exists(self):
        # Tested in setUp, but can add an explicit check for a new dir
        new_dir = os.path.join(self.test_dir, "new_sub_dir")
        self.assertFalse(os.path.exists(new_dir))
        FileSystemUtils.ensure_directory_exists(new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_ensure_directory_exists_already_exists(self):
        # This is already covered by setUp creating self.pdf_dir and self.image_dir
        # Just ensure it doesn't raise an error
        FileSystemUtils.ensure_directory_exists(self.pdf_dir)
        self.assertTrue(os.path.exists(self.pdf_dir))

    def test_get_file_name_without_extension(self):
        self.assertEqual(FileSystemUtils.get_file_name_without_extension("path/to/file.txt"), "file")
        self.assertEqual(FileSystemUtils.get_file_name_without_extension("file.pdf"), "file")
        self.assertEqual(FileSystemUtils.get_file_name_without_extension("no_extension"), "no_extension")

    def test_get_file_extension(self):
        self.assertEqual(FileSystemUtils.get_file_extension("path/to/file.txt"), ".txt")
        self.assertEqual(FileSystemUtils.get_file_extension("file.PDF"), ".PDF")
        self.assertEqual(FileSystemUtils.get_file_extension("no_extension"), "")

    def test_is_pdf(self):
        self.assertTrue(FileSystemUtils.is_pdf("document.pdf"))
        self.assertTrue(FileSystemUtils.is_pdf("document.PDF"))
        self.assertFalse(FileSystemUtils.is_pdf("document.txt"))
        self.assertFalse(FileSystemUtils.is_pdf("document.png"))

    def test_is_image(self):
        self.assertTrue(FileSystemUtils.is_image("image.png"))
        self.assertTrue(FileSystemUtils.is_image("photo.JPG"))
        self.assertFalse(FileSystemUtils.is_image("document.pdf"))
        self.assertFalse(FileSystemUtils.is_image("text.txt"))

    def test_is_text_file(self):
        self.assertTrue(FileSystemUtils.is_text_file("log.txt"))
        self.assertTrue(FileSystemUtils.is_text_file("data.csv"))
        self.assertFalse(FileSystemUtils.is_text_file("image.png"))
        self.assertFalse(FileSystemUtils.is_text_file("document.pdf"))

    def test_get_all_files_in_directory(self):
        files = FileSystemUtils.get_all_files_in_directory(self.pdf_dir)
        self.assertEqual(len(files), 3) # test1.pdf, test2.PDF, not_a_pdf.txt
        self.assertIn(os.path.join(self.pdf_dir, "test1.pdf"), files)
        self.assertIn(os.path.join(self.pdf_dir, "test2.PDF"), files)
        self.assertIn(os.path.join(self.pdf_dir, "not_a_pdf.txt"), files)

        pdf_files = FileSystemUtils.get_all_files_in_directory(self.pdf_dir, extension=".pdf")
        self.assertEqual(len(pdf_files), 2)
        self.assertIn(os.path.join(self.pdf_dir, "test1.pdf"), pdf_files)
        self.assertIn(os.path.join(self.pdf_dir, "test2.PDF"), pdf_files) # Case-insensitive check

    def test_list_pdf_files(self):
        pdf_files = FileSystemUtils.list_pdf_files(self.pdf_dir)
        self.assertEqual(len(pdf_files), 2)
        self.assertIn(os.path.join(self.pdf_dir, "test1.pdf"), pdf_files)
        self.assertIn(os.path.join(self.pdf_dir, "test2.PDF"), pdf_files)

    # Add tests for get_relative_path, get_absolute_path, and preprocess_image later
