"""
Page content analyzer for extracting context from PyTorch documentation pages.
"""
import re
from typing import Dict, List, Optional, Set, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import structlog

logger = structlog.get_logger(__name__)


class PageAnalyzer:
    """Analyzes PyTorch documentation pages to extract relevant context."""
    
    def __init__(self):
        # PyTorch-specific patterns
        self.module_patterns = [
            r'torch\.(\w+)',
            r'nn\.(\w+)',
            r'optim\.(\w+)',
            r'utils\.(\w+)',
            r'vision\.(\w+)',
            r'audio\.(\w+)'
        ]
        
        self.function_patterns = [
            r'def\s+(\w+)\s*\(',
            r'(\w+)\s*\([^)]*\)',
            r'torch\.(\w+)\s*\(',
            r'nn\.(\w+)\s*\('
        ]
        
        self.class_patterns = [
            r'class\s+(\w+)\s*\(',
            r'(\w+)\([^)]*\)\s*:',
            r'nn\.(\w+)\(',
            r'optim\.(\w+)\('
        ]
        
        # Code block patterns
        self.code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'<code>(.*?)</code>',
            r'<pre>(.*?)</pre>'
        ]
        
        logger.info("Page analyzer initialized")
    
    def analyze_page(self, page_content: str, page_url: str = "") -> Dict[str, Any]:
        """Analyze a page and extract relevant context."""
        try:
            # Parse HTML content
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract basic page information
            page_info = self._extract_page_info(soup, page_url)
            
            # Extract PyTorch entities
            entities = self._extract_pytorch_entities(page_content)
            
            # Extract code examples
            code_examples = self._extract_code_examples(soup, page_content)
            
            # Extract headings and structure
            structure = self._extract_page_structure(soup)
            
            # Extract current context (what the user is likely viewing)
            current_context = self._extract_current_context(soup, page_url)
            
            return {
                "page_info": page_info,
                "entities": entities,
                "code_examples": code_examples,
                "structure": structure,
                "current_context": current_context,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing page: {e}")
            return {"error": str(e)}
    
    def _extract_page_info(self, soup: BeautifulSoup, page_url: str) -> Dict[str, Any]:
        """Extract basic page information."""
        info = {
            "url": page_url,
            "title": "",
            "description": "",
            "breadcrumbs": [],
            "module_path": ""
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            info["title"] = title_tag.get_text().strip()
        
        # Extract description from meta tags
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            info["description"] = meta_desc.get('content', '').strip()
        
        # Extract breadcrumbs
        breadcrumb_elements = soup.find_all(['nav', 'ol', 'ul'], class_=re.compile(r'breadcrumb', re.I))
        if breadcrumb_elements:
            for element in breadcrumb_elements:
                links = element.find_all('a')
                for link in links:
                    info["breadcrumbs"].append(link.get_text().strip())
        
        # Extract module path from URL or title
        if page_url:
            info["module_path"] = self._extract_module_from_url(page_url)
        elif info["title"]:
            info["module_path"] = self._extract_module_from_title(info["title"])
        
        return info
    
    def _extract_pytorch_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract PyTorch entities from page content."""
        entities = {
            "modules": set(),
            "functions": set(),
            "classes": set(),
            "parameters": set(),
            "imports": set()
        }
        
        # Extract modules
        for pattern in self.module_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities["modules"].update(matches)
        
        # Extract functions
        for pattern in self.function_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities["functions"].update(matches)
        
        # Extract classes
        for pattern in self.class_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities["classes"].update(matches)
        
        # Extract parameters from function signatures
        param_pattern = r'(\w+)\s*[:=]\s*[^,)]+'
        param_matches = re.findall(param_pattern, content)
        entities["parameters"].update(param_matches)
        
        # Extract imports
        import_pattern = r'import\s+([^\s\n]+)'
        import_matches = re.findall(import_pattern, content)
        entities["imports"].update(import_matches)
        
        # Convert sets to lists for JSON serialization
        return {key: list(value) for key, value in entities.items()}
    
    def _extract_code_examples(self, soup: BeautifulSoup, content: str) -> List[Dict[str, Any]]:
        """Extract code examples from the page."""
        code_examples = []
        
        # Extract from HTML code blocks
        code_blocks = soup.find_all(['pre', 'code'])
        for i, block in enumerate(code_blocks):
            code_text = block.get_text().strip()
            if len(code_text) > 20:  # Filter out short snippets
                code_examples.append({
                    "index": i,
                    "code": code_text,
                    "type": "html_block",
                    "language": self._detect_code_language(code_text)
                })
        
        # Extract from markdown-style code blocks in content
        for pattern in self.code_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for i, match in enumerate(matches):
                if len(match.strip()) > 20:
                    code_examples.append({
                        "index": len(code_examples),
                        "code": match.strip(),
                        "type": "markdown_block",
                        "language": "python"
                    })
        
        return code_examples
    
    def _extract_page_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract page structure and headings."""
        structure = {
            "headings": [],
            "sections": [],
            "navigation": []
        }
        
        # Extract headings
        for level in range(1, 7):
            headings = soup.find_all(f'h{level}')
            for heading in headings:
                structure["headings"].append({
                    "level": level,
                    "text": heading.get_text().strip(),
                    "id": heading.get('id', '')
                })
        
        # Extract navigation elements
        nav_elements = soup.find_all(['nav', 'ul'], class_=re.compile(r'nav|menu', re.I))
        for nav in nav_elements:
            links = nav.find_all('a')
            nav_items = []
            for link in links:
                nav_items.append({
                    "text": link.get_text().strip(),
                    "href": link.get('href', ''),
                    "active": "active" in link.get('class', [])
                })
            structure["navigation"].append(nav_items)
        
        return structure
    
    def _extract_current_context(self, soup: BeautifulSoup, page_url: str) -> Dict[str, Any]:
        """Extract the current context (what user is likely viewing)."""
        context = {
            "current_function": "",
            "current_class": "",
            "current_module": "",
            "active_section": "",
            "highlighted_code": ""
        }
        
        # Try to identify current function/class from URL
        if page_url:
            context["current_module"] = self._extract_module_from_url(page_url)
            context["current_function"] = self._extract_function_from_url(page_url)
            context["current_class"] = self._extract_class_from_url(page_url)
        
        # Look for active navigation items
        active_nav = soup.find('a', class_=re.compile(r'active|current', re.I))
        if active_nav:
            context["active_section"] = active_nav.get_text().strip()
        
        # Look for highlighted or focused elements
        highlighted = soup.find(['code', 'pre'], class_=re.compile(r'highlight|focus|current', re.I))
        if highlighted:
            context["highlighted_code"] = highlighted.get_text().strip()
        
        return context
    
    def _extract_module_from_url(self, url: str) -> str:
        """Extract module name from URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        # Look for torch-related paths
        for part in path_parts:
            if 'torch' in part.lower() or part in ['nn', 'optim', 'utils', 'vision', 'audio']:
                return part
        
        return ""
    
    def _extract_function_from_url(self, url: str) -> str:
        """Extract function name from URL."""
        # Common patterns in PyTorch docs
        patterns = [
            r'/(\w+)\.html',
            r'#(\w+)',
            r'/(\w+)#'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_class_from_url(self, url: str) -> str:
        """Extract class name from URL."""
        # Similar to function extraction but look for class patterns
        patterns = [
            r'/(\w+)\.html',
            r'#(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match and match.group(1)[0].isupper():  # Classes typically start with uppercase
                return match.group(1)
        
        return ""
    
    def _extract_module_from_title(self, title: str) -> str:
        """Extract module name from page title."""
        # Look for torch.module patterns in title
        patterns = [
            r'torch\.(\w+)',
            r'(\w+) — PyTorch',
            r'PyTorch (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _detect_code_language(self, code: str) -> str:
        """Detect the programming language of code."""
        # Simple heuristics for language detection
        if any(keyword in code for keyword in ['import torch', 'torch.', 'nn.', 'optim.']):
            return "python"
        elif any(keyword in code for keyword in ['def ', 'class ', 'import ']):
            return "python"
        elif any(keyword in code for keyword in ['function ', 'var ', 'let ']):
            return "javascript"
        else:
            return "text"
    
    def create_enriched_query_context(self, query: str, page_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create enriched query context combining user query with page context."""
        context = {
            "original_query": query,
            "page_context": page_analysis.get("current_context", {}),
            "relevant_entities": [],
            "suggested_enhancements": [],
            "context_confidence": 0.0
        }
        
        # Extract entities from query
        query_entities = self._extract_pytorch_entities(query)
        
        # Find overlap with page entities
        page_entities = page_analysis.get("entities", {})
        
        # Calculate context relevance
        overlap_score = 0
        total_entities = 0
        
        for entity_type in ["modules", "functions", "classes"]:
            query_entity_set = set(query_entities.get(entity_type, []))
            page_entity_set = set(page_entities.get(entity_type, []))
            
            overlap = query_entity_set.intersection(page_entity_set)
            if overlap:
                context["relevant_entities"].extend(list(overlap))
                overlap_score += len(overlap)
            
            total_entities += len(query_entity_set) + len(page_entity_set)
        
        # Calculate context confidence
        if total_entities > 0:
            context["context_confidence"] = overlap_score / total_entities
        
        # Generate suggestions for query enhancement
        if context["context_confidence"] < 0.3:
            context["suggested_enhancements"] = self._generate_query_suggestions(
                query, page_entities, context["page_context"]
            )
        
        return context
    
    def _generate_query_suggestions(self, query: str, page_entities: Dict[str, List[str]], 
                                  page_context: Dict[str, Any]) -> List[str]:
        """Generate suggestions for enhancing the query with page context."""
        suggestions = []
        
        # Suggest adding current module context
        current_module = page_context.get("current_module", "")
        if current_module and current_module not in query.lower():
            suggestions.append(f"Consider specifying the module context: 'torch.{current_module}'")
        
        # Suggest adding current function context
        current_function = page_context.get("current_function", "")
        if current_function and current_function not in query.lower():
            suggestions.append(f"Are you asking about '{current_function}' specifically?")
        
        # Suggest related entities from the page
        page_modules = page_entities.get("modules", [])
        if page_modules and len(page_modules) > 0:
            suggestions.append(f"Related modules on this page: {', '.join(page_modules[:3])}")
        
        return suggestions


def main():
    """Test the page analyzer."""
    analyzer = PageAnalyzer()
    
    # Sample PyTorch documentation HTML
    sample_html = """
    <html>
    <head>
        <title>torch.tensor — PyTorch 2.1 documentation</title>
        <meta name="description" content="Creates a tensor from data">
    </head>
    <body>
        <nav class="breadcrumb">
            <a href="/docs/">Docs</a> > <a href="/docs/torch.html">torch</a> > <a href="#" class="active">tensor</a>
        </nav>
        <h1>torch.tensor</h1>
        <p>Creates a tensor from data</p>
        <h2>Parameters</h2>
        <ul>
            <li>data: Input data</li>
            <li>dtype: Data type</li>
        </ul>
        <h2>Examples</h2>
        <pre><code class="python">
import torch
data = [1, 2, 3, 4]
tensor = torch.tensor(data)
print(tensor)
        </code></pre>
    </body>
    </html>
    """
    
    # Analyze the page
    analysis = analyzer.analyze_page(sample_html, "https://pytorch.org/docs/stable/torch.html#torch.tensor")
    
    print("Page Analysis Results:")
    print(f"Title: {analysis['page_info']['title']}")
    print(f"Module: {analysis['page_info']['module_path']}")
    print(f"Entities: {analysis['entities']}")
    print(f"Code Examples: {len(analysis['code_examples'])}")
    print(f"Current Context: {analysis['current_context']}")
    
    # Test enriched query context
    query = "how to create a tensor"
    enriched_context = analyzer.create_enriched_query_context(query, analysis)
    print(f"\nEnriched Query Context:")
    print(f"Context Confidence: {enriched_context['context_confidence']:.2f}")
    print(f"Relevant Entities: {enriched_context['relevant_entities']}")


if __name__ == "__main__":
    import time
    main()
