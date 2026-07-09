import unittest
from pathlib import Path


class ProjectDocsTest(unittest.TestCase):
    def test_readme_documents_dataset_location(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("data/train.csv", readme)
        self.assertIn("data/test.csv", readme)
        self.assertIn("Kaggle", readme)

    def test_data_readme_exists(self):
        self.assertTrue(Path("data/README.md").exists())


if __name__ == "__main__":
    unittest.main()
