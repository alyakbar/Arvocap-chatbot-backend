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
import hashlib
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from urllib import robotparser
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import base64
from PIL import Image
import io
from config import USER_AGENT, REQUEST_DELAY, MAX_RETRIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.driver = None
        # robots.txt cache per domain
        self._robots_cache: Dict[str, Tuple[robotparser.RobotFileParser, Optional[float]]] = {}
        # seen content hashes to deduplicate
        self._seen_hashes: Set[str] = set()
        # seen URLs to avoid revisiting
        self._seen_urls: Set[str] = set()
        # optional dynamic crawl delay from robots
        self._default_delay = max(0.1, REQUEST_DELAY)
        # comprehensive crawling state
        self._crawl_queue: List[str] = []
        self._crawled_pages: List[Dict] = []
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
        """Scrape a single URL and extract content with metadata"""
        try:
            if self.use_selenium:
                return self._scrape_with_selenium(url)
            else:
                return self._scrape_with_requests(url)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return {"url": url, "content": "", "error": str(e)}
    
    def _scrape_with_requests(self, url: str) -> Dict:
        """Scrape using requests library; fallback to Selenium if enabled and content is thin"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                content = self._extract_content(soup)
                title = self._extract_title(soup)
                meta_desc = self._extract_meta_description(soup)
                canonical = self._extract_canonical(soup, url)
                headings = self._extract_headings(soup)
                pdf_links = self._extract_pdf_links(soup, url)
                images = self._extract_image_metadata(soup, url)
                
                result = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "links": self._extract_links(soup, url),
                    "meta_description": meta_desc,
                    "canonical": canonical,
                    "headings": headings,
                    "pdf_links": pdf_links,
                    "images": images,
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "word_count": len(content.split()) if content else 0,
                }

                # If content is too small and Selenium is available, try JS rendering once
                if (not content or len(content) < 400) and self.use_selenium and self.driver is not None:
                    logger.info("Thin content via requests, retrying with Selenium...")
                    return self._scrape_with_selenium(url)

                return result
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(self._default_delay * (attempt + 1))
    
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
        title = self._extract_title(soup) or (self.driver.title or "")
        meta_desc = self._extract_meta_description(soup)
        canonical = self._extract_canonical(soup, url)
        headings = self._extract_headings(soup)
        pdf_links = self._extract_pdf_links(soup, url)
        images = self._extract_image_metadata(soup, url)

        return {
            "url": url,
            "title": title,
            "content": content,
            "links": self._extract_links(soup, url),
            "meta_description": meta_desc,
            "canonical": canonical,
            "headings": headings,
            "pdf_links": pdf_links,
            "images": images,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "word_count": len(content.split()) if content else 0,
            "rendered": True,
        }
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text content from HTML"""
        # Remove non-content elements
        for script in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]):
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
                # Preserve paragraph and heading boundaries minimally
                parts = []
                for elem in elements:
                    # Remove nav/ul lists that are menus
                    for nav in elem.select('nav, .menu, .navbar, .breadcrumb'):
                        nav.decompose()
                    parts.append(elem.get_text(separator='\n'))
                content_text = "\n".join(parts)
                break
        
        # Fallback to body content
        if not content_text:
            content_text = soup.get_text(separator='\n')
        
        # Clean up text
        # Collapse excessive blank lines but keep some structure
        content_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', content_text)
        content_text = re.sub(r'[ \t\x0b\x0c\r]+', ' ', content_text)
        content_text = content_text.strip()
        return content_text

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        og = soup.find('meta', property='og:title')
        if og and og.get('content'):
            return og['content'].strip()
        h1 = soup.find('h1')
        return h1.get_text(strip=True) if h1 else ""
    
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
        """Extract meta description (standard and OpenGraph)"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content', '').strip()
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc.get('content', '').strip()
        return ""

    def _extract_canonical(self, soup: BeautifulSoup, base_url: str) -> str:
        link = soup.find('link', rel='canonical')
        href = link.get('href') if link else ''
        return self._normalize_url(urljoin(base_url, href)) if href else ''

    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        headings = []
        for tag in ['h1', 'h2', 'h3']:
            for h in soup.find_all(tag):
                txt = h.get_text(strip=True)
                if txt:
                    headings.append(f"{tag.upper()}: {txt}")
        return headings[:20]

    def _extract_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        pdfs = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full = urljoin(base_url, href)
            if full.lower().endswith('.pdf'):
                pdfs.append(self._normalize_url(full))
        return list(dict.fromkeys(pdfs))

    def _extract_image_metadata(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract comprehensive image metadata including alt text, dimensions, and descriptions"""
        images = []
        
        for img in soup.find_all('img'):
            try:
                src = img.get('src')
                if not src:
                    continue
                
                # Build absolute URL
                img_url = urljoin(base_url, src)
                
                # Extract metadata
                img_data = {
                    'url': img_url,
                    'alt': img.get('alt', '').strip(),
                    'title': img.get('title', '').strip(),
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'class': ' '.join(img.get('class', [])),
                    'id': img.get('id', ''),
                    'data_attributes': {}
                }
                
                # Extract data-* attributes
                for attr in img.attrs:
                    if attr.startswith('data-'):
                        img_data['data_attributes'][attr] = img.attrs[attr]
                
                # Try to get image dimensions if not specified
                if not img_data['width'] or not img_data['height']:
                    try:
                        # Get image info (HEAD request to avoid downloading full image)
                        response = self.session.head(img_url, timeout=5)
                        if response.status_code == 200:
                            content_length = response.headers.get('content-length')
                            if content_length:
                                img_data['file_size'] = int(content_length)
                        
                        # If that doesn't work, try a small range request
                        if 'file_size' not in img_data:
                            response = self.session.get(
                                img_url, 
                                headers={'Range': 'bytes=0-2048'}, 
                                timeout=5
                            )
                            if response.status_code in [200, 206]:
                                try:
                                    img_obj = Image.open(io.BytesIO(response.content))
                                    img_data['width'] = str(img_obj.width)
                                    img_data['height'] = str(img_obj.height)
                                    img_data['format'] = img_obj.format
                                except:
                                    pass
                    except:
                        pass
                
                # Extract surrounding context (figure, caption, etc.)
                parent = img.parent
                if parent:
                    if parent.name == 'figure':
                        caption = parent.find('figcaption')
                        if caption:
                            img_data['caption'] = caption.get_text().strip()
                    
                    # Look for nearby text that might describe the image
                    prev_text = []
                    next_text = []
                    
                    # Get previous sibling text
                    prev_sibling = img.previous_sibling
                    while prev_sibling and len(prev_text) < 50:
                        if hasattr(prev_sibling, 'get_text'):
                            text = prev_sibling.get_text().strip()
                            if text:
                                prev_text.insert(0, text)
                                break
                        elif isinstance(prev_sibling, str):
                            text = prev_sibling.strip()
                            if text:
                                prev_text.insert(0, text)
                                break
                        prev_sibling = prev_sibling.previous_sibling
                    
                    # Get next sibling text
                    next_sibling = img.next_sibling
                    while next_sibling and len(next_text) < 50:
                        if hasattr(next_sibling, 'get_text'):
                            text = next_sibling.get_text().strip()
                            if text:
                                next_text.append(text)
                                break
                        elif isinstance(next_sibling, str):
                            text = next_sibling.strip()
                            if text:
                                next_text.append(text)
                                break
                        next_sibling = next_sibling.next_sibling
                    
                    if prev_text:
                        img_data['context_before'] = ' '.join(prev_text)
                    if next_text:
                        img_data['context_after'] = ' '.join(next_text)
                
                # Skip if no meaningful metadata
                if not any([img_data['alt'], img_data['title'], img_data.get('caption')]):
                    continue
                
                images.append(img_data)
                
            except Exception as e:
                logger.warning(f"Error extracting image metadata: {e}")
                continue
        
        return images
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not a binary/file download"""
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

    def _normalize_url(self, url: str) -> str:
        """Normalize URLs: remove fragments, trim tracking params, unify trailing slashes"""
        try:
            parsed = urlparse(url)
            # Remove fragment
            fragmentless = parsed._replace(fragment='')
            # Drop common tracking query params
            q = [(k, v) for k, v in parse_qsl(fragmentless.query, keep_blank_values=False) if not k.lower().startswith('utm_')]
            query = urlencode(q, doseq=True)
            # Normalize path (remove double slashes)
            path = re.sub(r'/+', '/', fragmentless.path)
            # Rebuild URL
            normalized = urlunparse((fragmentless.scheme, fragmentless.netloc, path, '', query, ''))
            # Remove trailing slash for consistency (except root)
            if normalized.endswith('/') and len(path) > 1:
                normalized = normalized[:-1]
            return normalized
        except Exception:
            return url

    def _robots_for_domain(self, base_url: str) -> Tuple[Optional[robotparser.RobotFileParser], Optional[float]]:
        domain = urlparse(base_url).netloc
        if domain in self._robots_cache:
            return self._robots_cache[domain]
        rp = robotparser.RobotFileParser()
        robots_url = urlunparse((urlparse(base_url).scheme, domain, '/robots.txt', '', '', ''))
        try:
            rp.set_url(robots_url)
            rp.read()
            # Attempt to parse crawl-delay (not exposed by API reliably)
            crawl_delay = None
            try:
                # crude parse of robots content
                txt = self.session.get(robots_url, timeout=5).text
                m = re.search(r'(?i)crawl-delay\s*:\s*(\d+(?:\.\d+)?)', txt)
                if m:
                    crawl_delay = float(m.group(1))
            except Exception:
                pass
            self._robots_cache[domain] = (rp, crawl_delay)
            return rp, crawl_delay
        except Exception:
            self._robots_cache[domain] = (None, None)
            return None, None

    def _allowed_by_robots(self, url: str) -> bool:
        try:
            rp, _ = self._robots_for_domain(url)
            if rp is None:
                return True
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def _discover_sitemap_urls(self, start_url: str, limit: int = 50) -> List[str]:
        """Fetch URLs from sitemap.xml when available (best-effort)."""
        urls: List[str] = []
        base = urlparse(start_url)
        sitemap_url = urlunparse((base.scheme, base.netloc, '/sitemap.xml', '', '', ''))
        try:
            resp = self.session.get(sitemap_url, timeout=10)
            if resp.status_code != 200 or 'xml' not in (resp.headers.get('Content-Type', '')).lower():
                return []
            root = ET.fromstring(resp.content)
            for loc in root.findall('.//{*}loc'):
                if loc.text:
                    u = self._normalize_url(loc.text.strip())
                    if self._is_valid_url(u):
                        urls.append(u)
                        if len(urls) >= limit:
                            break
        except Exception:
            return []
        return urls
    
    def scrape_website(
        self,
        start_url: str,
        max_pages: int = 200,  # Increased default for full site scraping
        same_domain_only: bool = True,
        max_depth: int = 5,  # Increased default depth
        use_sitemaps: bool = True,
        respect_robots: bool = True,
        comprehensive_crawl: bool = True,  # New parameter for full site discovery
    ) -> List[Dict]:
        """
        Scrape multiple pages from a website with comprehensive discovery.
        Enhanced to find ALL pages on a website through multiple discovery methods.
        """
        scraped_data: List[Dict] = []
        visited_urls: Set[str] = set()
        queue: List[Tuple[str, int]] = []

        start = self._normalize_url(start_url)
        queue.append((start, 0))
        domain = urlparse(start).netloc

        # Multiple discovery methods for comprehensive crawling
        if comprehensive_crawl:
            # 1. Sitemap discovery (multiple common sitemap locations)
            if use_sitemaps:
                sitemap_urls = self._discover_comprehensive_sitemaps(start, max_pages)
                for su in sitemap_urls:
                    if same_domain_only and urlparse(su).netloc != domain:
                        continue
                    queue.append((su, 0))
            
            # 2. Common page discovery (robots.txt, common paths)
            common_paths = self._discover_common_paths(start, domain)
            for path_url in common_paths:
                if same_domain_only and urlparse(path_url).netloc != domain:
                    continue
                queue.append((path_url, 0))

        # Determine polite delay (robots crawl-delay if present)
        delay = self._default_delay
        if respect_robots:
            _, cd = self._robots_for_domain(start)
            if cd is not None:
                delay = max(delay, cd)

        processed_count = 0
        while queue and len(scraped_data) < max_pages:
            url, depth = queue.pop(0)
            url = self._normalize_url(url)
            
            if url in visited_urls:
                continue
            visited_urls.add(url)

            # Domain filter
            if same_domain_only and urlparse(url).netloc != domain:
                continue

            # robots.txt check
            if respect_robots and not self._allowed_by_robots(url):
                logger.info(f"Skipping disallowed by robots.txt: {url}")
                continue

            logger.info(f"Scraping (depth {depth}, {len(scraped_data)}/{max_pages}): {url}")
            page_data = self.scrape_url(url)

            content = (page_data.get('content') or '').strip()
            if content:
                # Deduplicate by content hash (avoid repeated templates)
                h = hashlib.sha256(content.encode('utf-8')).hexdigest()
                if h in self._seen_hashes:
                    logger.debug("Duplicate content hash; skipping add")
                else:
                    self._seen_hashes.add(h)
                    
                    # Enhance page data with comprehensive metadata
                    page_data['crawl_depth'] = depth
                    page_data['discovered_images_count'] = len(page_data.get('images', []))
                    page_data['discovered_links_count'] = len(page_data.get('links', []))
                    
                    scraped_data.append(page_data)

                # Enhanced link discovery for comprehensive crawling
                if depth < max_depth:
                    # Regular links
                    for link in page_data.get('links', []) or []:
                        if not self._is_valid_url(link):
                            continue
                        nlink = self._normalize_url(link)
                        if nlink not in visited_urls and all(nlink != u for u, _ in queue):
                            queue.append((nlink, depth + 1))
                    
                    # Enhanced link discovery if comprehensive crawl is enabled
                    if comprehensive_crawl:
                        additional_links = self._discover_additional_links(page_data, url, domain)
                        for link in additional_links:
                            nlink = self._normalize_url(link)
                            if nlink not in visited_urls and all(nlink != u for u, _ in queue):
                                queue.append((nlink, depth + 1))

            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"Progress: {len(scraped_data)} pages scraped, {len(queue)} in queue")

            time.sleep(delay)

        logger.info(f"Scraping completed: {len(scraped_data)} unique pages found")
        return scraped_data

    def _discover_comprehensive_sitemaps(self, base_url: str, max_urls: int = 1000) -> List[str]:
        """Discover URLs from multiple sitemap locations and formats"""
        urls = []
        domain_url = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
        
        # Common sitemap locations
        sitemap_locations = [
            f"{domain_url}/sitemap.xml",
            f"{domain_url}/sitemap_index.xml", 
            f"{domain_url}/sitemaps.xml",
            f"{domain_url}/sitemap/index.xml",
            f"{domain_url}/wp-sitemap.xml",  # WordPress
            f"{domain_url}/page-sitemap.xml",
            f"{domain_url}/post-sitemap.xml",
        ]
        
        for sitemap_url in sitemap_locations:
            try:
                response = self.session.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    # Parse XML sitemap
                    try:
                        root = ET.fromstring(response.content)
                        # Handle different sitemap namespaces
                        for url_elem in root.iter():
                            if url_elem.tag.endswith('}loc') or url_elem.tag == 'loc':
                                url = url_elem.text
                                if url and len(urls) < max_urls:
                                    urls.append(url)
                    except ET.ParseError:
                        # Try parsing as sitemap index
                        for line in response.text.split('\n'):
                            if '<loc>' in line and '</loc>' in line:
                                url = line.split('<loc>')[1].split('</loc>')[0]
                                if url and len(urls) < max_urls:
                                    urls.append(url)
            except:
                continue
        
        return list(dict.fromkeys(urls))  # Remove duplicates

    def _discover_common_paths(self, base_url: str, domain: str) -> List[str]:
        """Discover common website paths that might exist"""
        base = f"{urlparse(base_url).scheme}://{domain}"
        common_paths = [
            "/about", "/about-us", "/contact", "/contact-us", "/services", 
            "/products", "/blog", "/news", "/events", "/team", "/careers",
            "/faq", "/help", "/support", "/privacy", "/terms", "/legal",
            "/portfolio", "/gallery", "/testimonials", "/reviews",
            "/search", "/archives", "/categories", "/tags",
            "/page/1", "/page/2", "/page/3",  # Pagination
        ]
        
        urls = []
        for path in common_paths:
            urls.append(f"{base}{path}")
            urls.append(f"{base}{path}/")  # With trailing slash
        
        return urls

    def _discover_additional_links(self, page_data: dict, current_url: str, domain: str) -> List[str]:
        """Discover additional links through pattern analysis"""
        additional_links = []
        
        # Look for pagination patterns in content
        content = page_data.get('content', '')
        links = page_data.get('links', [])
        
        # Extract numbered pagination links
        for link in links:
            if re.search(r'/page/\d+|page=\d+|p=\d+', link, re.IGNORECASE):
                additional_links.append(link)
        
        # Look for "next page" or "more" links
        for link in links:
            if re.search(r'next|more|continue|page.*\d', link, re.IGNORECASE):
                additional_links.append(link)
        
        # Generate potential pagination URLs based on current URL
        parsed = urlparse(current_url)
        if '/page/' in parsed.path or 'page=' in parsed.query:
            # Try next few pages
            for i in range(2, 6):  # Check pages 2-5
                if '/page/' in parsed.path:
                    new_path = re.sub(r'/page/\d+', f'/page/{i}', parsed.path)
                    additional_links.append(f"{parsed.scheme}://{parsed.netloc}{new_path}")
                elif 'page=' in parsed.query:
                    new_query = re.sub(r'page=\d+', f'page={i}', parsed.query)
                    additional_links.append(f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}")
        
        return additional_links
    
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
