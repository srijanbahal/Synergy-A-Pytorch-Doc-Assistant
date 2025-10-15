"""
PyTorch documentation scraper for collecting and processing PyTorch docs.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
import structlog
from backend.config import settings

logger = structlog.get_logger(__name__)


class PyTorchScraper:
    """Scraper for PyTorch documentation with caching and incremental updates."""
    
    def __init__(self):
        self.base_url = "https://pytorch.org/docs/stable/"
        self.session: Optional[aiohttp.ClientSession] = None
        self.scraped_urls: Set[str] = set()
        self.cache_file = Path(settings.cache_dir) / "scraped_urls.json"
        self.docs_dir = Path(settings.pytorch_docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load previously scraped URLs
        self._load_scraped_urls()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": settings.user_agent},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _load_scraped_urls(self):
        """Load previously scraped URLs from cache."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.scraped_urls = set(json.load(f))
                logger.info(f"Loaded {len(self.scraped_urls)} cached URLs")
            except Exception as e:
                logger.warning(f"Failed to load cached URLs: {e}")
    
    def _save_scraped_urls(self):
        """Save scraped URLs to cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(list(self.scraped_urls), f, indent=2)
            logger.info(f"Saved {len(self.scraped_urls)} URLs to cache")
        except Exception as e:
            logger.error(f"Failed to save URLs to cache: {e}")
    
    def _get_file_path(self, url: str) -> Path:
        """Get file path for storing scraped content."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        if not path.endswith('.html'):
            path += '.html'
        return self.docs_dir / path
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract metadata from the page."""
        metadata = {
            'url': url,
            'title': '',
            'module': '',
            'function': '',
            'class': '',
            'description': '',
            'scraped_at': time.time()
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract module path from URL
        url_parts = url.replace(self.base_url, '').split('/')
        if url_parts:
            metadata['module'] = url_parts[0]
        
        # Extract function/class name from title or headings
        h1 = soup.find('h1')
        if h1:
            h1_text = h1.get_text().strip()
            if '.' in h1_text:
                parts = h1_text.split('.')
                if len(parts) >= 2:
                    metadata['class'] = parts[-2] if parts[-2] else ''
                    metadata['function'] = parts[-1] if parts[-1] else ''
        
        # Extract description from meta description or first paragraph
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '').strip()
        else:
            first_p = soup.find('p')
            if first_p:
                metadata['description'] = first_p.get_text().strip()[:200]
        
        return metadata
    
    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract code examples from the page."""
        code_examples = []
        
        # Look for code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        
        for i, block in enumerate(code_blocks):
            if block.name == 'pre':
                code_text = block.get_text().strip()
                if code_text and len(code_text) > 10:  # Filter out short snippets
                    code_examples.append({
                        'type': 'code_block',
                        'code': code_text,
                        'index': i
                    })
            elif block.name == 'code' and block.parent.name != 'pre':
                code_text = block.get_text().strip()
                if code_text and len(code_text) > 5:
                    code_examples.append({
                        'type': 'inline_code',
                        'code': code_text,
                        'index': i
                    })
        
        return code_examples
    
    def _extract_navigation_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract navigation links to other documentation pages."""
        links = []
        
        # Find all links to PyTorch documentation
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = urljoin(self.base_url, href)
            elif href.startswith('./') or not href.startswith('http'):
                href = urljoin(current_url, href)
            
            # Filter for PyTorch docs URLs
            if href.startswith(self.base_url) and href not in links:
                links.append(href)
        
        return links
    
    async def scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape a single page and return structured data."""
        if url in self.scraped_urls:
            logger.debug(f"URL already scraped: {url}")
            return None
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract content
                metadata = self._extract_metadata(soup, url)
                code_examples = self._extract_code_examples(soup)
                navigation_links = self._extract_navigation_links(soup, url)
                
                # Get main content (remove navigation, sidebar, etc.)
                main_content = soup.find('main') or soup.find('div', class_='main-content')
                if not main_content:
                    main_content = soup.find('body')
                
                # Remove unwanted elements
                for element in main_content.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                    element.decompose()
                
                content = {
                    'metadata': metadata,
                    'content': main_content.get_text().strip() if main_content else '',
                    'html_content': str(main_content) if main_content else '',
                    'code_examples': code_examples,
                    'navigation_links': navigation_links
                }
                
                # Save to file
                file_path = self._get_file_path(url)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
                
                self.scraped_urls.add(url)
                logger.info(f"Scraped and saved: {url}")
                
                # Add delay to be respectful
                await asyncio.sleep(settings.scraping_delay)
                
                return content
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    async def discover_pytorch_urls(self) -> List[str]:
        """Discover PyTorch documentation URLs by crawling the main index."""
        urls = set()
        
        try:
            # Start with main documentation index
            index_urls = [
                "https://pytorch.org/docs/stable/index.html",
                "https://pytorch.org/docs/stable/torch.html",
                "https://pytorch.org/docs/stable/torchvision.html",
                "https://pytorch.org/docs/stable/torchaudio.html",
            ]
            
            for url in index_urls:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract all documentation links
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                
                                # Convert to absolute URL
                                if href.startswith('/'):
                                    href = urljoin(self.base_url, href)
                                elif not href.startswith('http'):
                                    href = urljoin(url, href)
                                
                                # Filter for PyTorch docs
                                if (href.startswith(self.base_url) and 
                                    not href.endswith(('.pdf', '.zip', '.tar.gz')) and
                                    '#' not in href):
                                    urls.add(href)
                            
                            await asyncio.sleep(settings.scraping_delay)
                            
                except Exception as e:
                    logger.error(f"Error discovering URLs from {url}: {e}")
            
            logger.info(f"Discovered {len(urls)} PyTorch documentation URLs")
            return list(urls)
            
        except Exception as e:
            logger.error(f"Error in URL discovery: {e}")
            return []
    
    async def scrape_all(self, max_pages: Optional[int] = None) -> Dict:
        """Scrape all PyTorch documentation pages."""
        logger.info("Starting PyTorch documentation scraping")
        
        # Discover URLs
        urls = await self.discover_pytorch_urls()
        
        if max_pages:
            urls = urls[:max_pages]
        
        logger.info(f"Scraping {len(urls)} pages")
        
        # Scrape pages with concurrency control
        semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_page(url)
        
        # Execute scraping
        results = await asyncio.gather(
            *[scrape_with_semaphore(url) for url in urls],
            return_exceptions=True
        )
        
        # Filter successful results
        successful = [r for r in results if isinstance(r, dict)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        # Save cache
        self._save_scraped_urls()
        
        summary = {
            'total_urls': len(urls),
            'successful_scrapes': len(successful),
            'failed_scrapes': len(failed),
            'already_scraped': len(urls) - len(successful) - len(failed)
        }
        
        logger.info(f"Scraping complete: {summary}")
        return summary


async def main():
    """Main function for running the scraper."""
    async with PyTorchScraper() as scraper:
        summary = await scraper.scrape_all()
        print(f"Scraping summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
