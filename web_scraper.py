import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import logging
from config import USER_AGENT, REQUEST_DELAY, MAX_RETRIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, use_selenium=False):
        self.use_selenium = use_selenium
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.driver = None
        
        if use_selenium:
            self.setup_selenium()
    
    def setup_selenium(self):
        """Setup Selenium WebDriver for JavaScript-heavy sites"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'--user-agent={USER_AGENT}')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {e}")
            self.use_selenium = False
    
    def scrape_url(self, url: str) -> Dict:
        """Scrape a single URL and extract content"""
        try:
            if self.use_selenium:
                return self._scrape_with_selenium(url)
            else:
                return self._scrape_with_requests(url)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {"url": url, "content": "", "error": str(e)}
    
    def _scrape_with_requests(self, url: str) -> Dict:
        """Scrape using requests library"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                content = self._extract_content(soup)
                
                return {
                    "url": url,
                    "title": soup.title.string if soup.title else "",
                    "content": content,
                    "links": self._extract_links(soup, url),
                    "meta_description": self._extract_meta_description(soup)
                }
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(REQUEST_DELAY * (attempt + 1))
    
    def _scrape_with_selenium(self, url: str) -> Dict:
        """Scrape using Selenium for JavaScript-heavy sites"""
        self.driver.get(url)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Wait for dynamic content to load
        time.sleep(2)
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = self._extract_content(soup)
        
        return {
            "url": url,
            "title": self.driver.title,
            "content": content,
            "links": self._extract_links(soup, url),
            "meta_description": self._extract_meta_description(soup)
        }
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Find main content areas
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.post', '.entry', '.page-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_text = " ".join([elem.get_text() for elem in elements])
                break
        
        # Fallback to body content
        if not content_text:
            content_text = soup.get_text()
        
        # Clean up text
        content_text = re.sub(r'\s+', ' ', content_text).strip()
        return content_text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            if self._is_valid_url(full_url):
                links.append(full_url)
        return list(set(links))  # Remove duplicates
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '')
        return ""
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not a file download"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Skip file downloads
            skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar']
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
                
            return True
        except:
            return False
    
    def scrape_website(self, start_url: str, max_pages: int = 50, 
                      same_domain_only: bool = True) -> List[Dict]:
        """Scrape multiple pages from a website"""
        scraped_data = []
        visited_urls = set()
        urls_to_visit = [start_url]
        
        domain = urlparse(start_url).netloc
        
        while urls_to_visit and len(scraped_data) < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in visited_urls:
                continue
                
            visited_urls.add(url)
            
            # Skip if different domain and same_domain_only is True
            if same_domain_only and urlparse(url).netloc != domain:
                continue
            
            logger.info(f"Scraping: {url}")
            page_data = self.scrape_url(url)
            
            if page_data.get('content'):
                scraped_data.append(page_data)
                
                # Add new links to visit
                if 'links' in page_data:
                    for link in page_data['links']:
                        if link not in visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)
            
            time.sleep(REQUEST_DELAY)
        
        return scraped_data
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

def main():
    """Example usage"""
    scraper = WebScraper(use_selenium=False)
    
    try:
        # Example: Scrape a website
        url = input("Enter the website URL to scrape: ")
        max_pages = int(input("Enter maximum pages to scrape (default 10): ") or "10")
        
        print(f"Starting to scrape {url}...")
        data = scraper.scrape_website(url, max_pages=max_pages)
        
        # Save scraped data
        output_file = "scraped_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Scraped {len(data)} pages. Data saved to {output_file}")
        
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
