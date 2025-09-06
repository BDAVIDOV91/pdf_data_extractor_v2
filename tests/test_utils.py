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

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ensure_directory_exists(self):
        self.assertFalse(os.path.exists(self.test_dir))
        FileSystemUtils.ensure_directory_exists(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

    def test_ensure_directory_exists_already_exists(self):
        os.makedirs(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        FileSystemUtils.ensure_directory_exists(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))

if __name__ == '__main__':
    unittest.main()