import unittest
import os
import sys

# Allow importing web_scraper.py directly from this folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from web_scraper import WebScraper

class TestWebScraperUtils(unittest.TestCase):
    def setUp(self):
        self.scraper = WebScraper(use_selenium=False)

    def test_normalize_url(self):
        u = 'https://example.com/path//to/page/?utm_source=newsletter#section'
        n = self.scraper._normalize_url(u)
        self.assertEqual(n, 'https://example.com/path/to/page')

    def test_is_valid_url(self):
        self.assertTrue(self.scraper._is_valid_url('https://example.com'))
        self.assertFalse(self.scraper._is_valid_url('mailto:test@example.com'))
        self.assertFalse(self.scraper._is_valid_url('https://example.com/file.pdf'))

if __name__ == '__main__':
    unittest.main()
