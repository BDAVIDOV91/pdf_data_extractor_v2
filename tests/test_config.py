import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings

class TestConfig(unittest.TestCase):

    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.input_dir, Path("input_pdfs"))
        self.assertEqual(settings.output_dir, Path("output_reports"))
        self.assertEqual(settings.log_file, Path("logs/extraction.log"))
        self.assertTrue(settings.enable_ocr)
        self.assertEqual(settings.tesseract_cmd, "/usr/bin/tesseract")

    def test_env_file_loading(self):
        with open(".env", "w") as f:
            f.write("INPUT_DIR=my_input_dir\n")
            f.write("ENABLE_OCR=True\n")
            f.write("OLLAMA_LLM_MODEL=test_ollama_model\n")
            f.write("HUGGINGFACE_EMBEDDINGS_MODEL=test_hf_model\n")
        
        settings = Settings()
        self.assertEqual(settings.input_dir, Path("my_input_dir"))
        self.assertTrue(settings.enable_ocr)
        self.assertEqual(settings.ollama_llm_model, "test_ollama_model")
        self.assertEqual(settings.huggingface_embeddings_model, "test_hf_model")

        import os
        os.remove(".env")

if __name__ == '__main__':
    unittest.main() __name__ == '__main__':
    unittest.main()