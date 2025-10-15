/**
 * PyTorch RAG Assistant - Content Script
 * Injected into PyTorch documentation pages
 */

class PyTorchContentScript {
    constructor() {
        this.isInjected = false;
        this.highlightedElements = [];
        this.init();
    }

    init() {
        // Only run on PyTorch documentation pages
        if (!window.location.hostname.includes('pytorch.org')) {
            return;
        }

        // Wait for page to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        if (this.isInjected) return;
        
        this.isInjected = true;
        
        // Inject CSS
        this.injectCSS();
        
        // Add PyTorch RAG button to the page
        this.addRAGButton();
        
        // Setup page analysis
        this.analyzePage();
        
        // Listen for messages from popup
        this.setupMessageListener();
        
        // Setup page change detection
        this.setupPageChangeDetection();
        
        console.log('PyTorch RAG Assistant content script loaded');
    }

    injectCSS() {
        const style = document.createElement('style');
        style.textContent = `
            .pytorch-rag-button {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                background: linear-gradient(135deg, #ee4c2c 0%, #ff6b35 100%);
                color: white;
                border: none;
                border-radius: 50px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(238, 76, 44, 0.3);
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .pytorch-rag-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(238, 76, 44, 0.4);
            }
            
            .pytorch-rag-button:active {
                transform: translateY(0);
            }
            
            .pytorch-rag-highlight {
                background: rgba(238, 76, 44, 0.2) !important;
                border-radius: 4px;
                transition: background 0.3s ease;
            }
            
            .pytorch-rag-highlight:hover {
                background: rgba(238, 76, 44, 0.3) !important;
            }
            
            .pytorch-rag-tooltip {
                position: absolute;
                background: #1e293b;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                z-index: 10001;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s ease;
                max-width: 200px;
                word-wrap: break-word;
            }
            
            .pytorch-rag-tooltip.show {
                opacity: 1;
            }
            
            .pytorch-rag-code-highlight {
                background: rgba(238, 76, 44, 0.1) !important;
                border-left: 3px solid #ee4c2c !important;
                padding-left: 8px !important;
            }
            
            .pytorch-rag-function-highlight {
                background: rgba(34, 197, 94, 0.1) !important;
                border-left: 3px solid #22c55e !important;
                padding-left: 8px !important;
            }
            
            .pytorch-rag-class-highlight {
                background: rgba(59, 130, 246, 0.1) !important;
                border-left: 3px solid #3b82f6 !important;
                padding-left: 8px !important;
            }
        `;
        document.head.appendChild(style);
    }

    addRAGButton() {
        // Check if button already exists
        if (document.querySelector('.pytorch-rag-button')) {
            return;
        }

        const button = document.createElement('button');
        button.className = 'pytorch-rag-button';
        button.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                <path d="M13 8H7"></path>
                <path d="M17 12H7"></path>
            </svg>
            PyTorch RAG Assistant
        `;
        
        button.addEventListener('click', () => {
            this.openRAGPopup();
        });
        
        document.body.appendChild(button);
        
        // Auto-hide button after 5 seconds, show on hover
        setTimeout(() => {
            if (!button.matches(':hover')) {
                button.style.opacity = '0.7';
                button.style.transform = 'scale(0.95)';
            }
        }, 5000);
        
        button.addEventListener('mouseenter', () => {
            button.style.opacity = '1';
            button.style.transform = 'scale(1)';
        });
        
        button.addEventListener('mouseleave', () => {
            setTimeout(() => {
                if (!button.matches(':hover')) {
                    button.style.opacity = '0.7';
                    button.style.transform = 'scale(0.95)';
                }
            }, 2000);
        });
    }

    analyzePage() {
        // Extract page context
        const pageContext = this.extractPageContext();
        
        // Send context to background script
        chrome.runtime.sendMessage({
            type: 'PAGE_ANALYZED',
            data: pageContext
        });
        
        // Highlight PyTorch entities
        this.highlightPyTorchEntities();
        
        // Add hover tooltips for code elements
        this.addCodeTooltips();
    }

    extractPageContext() {
        const context = {
            url: window.location.href,
            title: document.title,
            module: this.extractModule(),
            function: this.extractCurrentFunction(),
            class: this.extractCurrentClass(),
            description: this.extractDescription(),
            codeExamples: this.extractCodeExamples(),
            headings: this.extractHeadings(),
            navigation: this.extractNavigation(),
            entities: this.extractPyTorchEntities()
        };
        
        return context;
    }

    extractModule() {
        // Extract module from URL or page content
        const urlMatch = window.location.pathname.match(/\/docs\/stable\/([^\/]+)/);
        if (urlMatch) {
            return urlMatch[1];
        }
        
        // Try to extract from page content
        const moduleElements = document.querySelectorAll('h1, .module-name, .api-header');
        for (const element of moduleElements) {
            const text = element.textContent.toLowerCase();
            if (text.includes('torch.') && !text.includes('function')) {
                const match = text.match(/torch\.(\w+)/);
                if (match) {
                    return match[1];
                }
            }
        }
        
        return null;
    }

    extractCurrentFunction() {
        // Look for function definitions in the page
        const functionElements = document.querySelectorAll('h1, h2, .function-name, .api-header');
        for (const element of functionElements) {
            const text = element.textContent;
            if (text.includes('torch.') && text.includes('(')) {
                const match = text.match(/torch\.\w+\.(\w+)/);
                if (match) {
                    return match[1];
                }
            }
        }
        
        return null;
    }

    extractCurrentClass() {
        // Look for class definitions
        const classElements = document.querySelectorAll('h1, h2, .class-name, .api-header');
        for (const element of classElements) {
            const text = element.textContent;
            if (text.includes('class ') || text.includes('torch.nn.')) {
                const match = text.match(/class\s+(\w+)|torch\.nn\.(\w+)/);
                if (match) {
                    return match[1] || match[2];
                }
            }
        }
        
        return null;
    }

    extractDescription() {
        // Extract main description from the page
        const descriptionSelectors = [
            '.module-docstring',
            '.class-docstring', 
            '.function-docstring',
            'p:first-of-type',
            '.description'
        ];
        
        for (const selector of descriptionSelectors) {
            const element = document.querySelector(selector);
            if (element && element.textContent.trim().length > 20) {
                return element.textContent.trim().substring(0, 200);
            }
        }
        
        return null;
    }

    extractCodeExamples() {
        const codeExamples = [];
        const codeBlocks = document.querySelectorAll('pre, code, .highlight, .codehilite');
        
        codeBlocks.forEach((block, index) => {
            const code = block.textContent.trim();
            if (code.length > 10 && code.includes('torch')) {
                codeExamples.push({
                    index: index,
                    code: code,
                    element: block
                });
            }
        });
        
        return codeExamples;
    }

    extractHeadings() {
        const headings = [];
        const headingElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        
        headingElements.forEach(heading => {
            headings.push({
                level: parseInt(heading.tagName.substring(1)),
                text: heading.textContent.trim(),
                id: heading.id || null
            });
        });
        
        return headings;
    }

    extractNavigation() {
        const navigation = [];
        const navElements = document.querySelectorAll('nav a, .navigation a, .sidebar a');
        
        navElements.forEach(link => {
            if (link.href && link.textContent.trim()) {
                navigation.push({
                    text: link.textContent.trim(),
                    href: link.href,
                    active: link.classList.contains('active') || link.classList.contains('current')
                });
            }
        });
        
        return navigation;
    }

    extractPyTorchEntities() {
        const entities = {
            modules: new Set(),
            functions: new Set(),
            classes: new Set(),
            parameters: new Set()
        };
        
        // Extract from page text
        const pageText = document.body.textContent;
        
        // Extract modules
        const moduleMatches = pageText.match(/torch\.(\w+)/g);
        if (moduleMatches) {
            moduleMatches.forEach(match => {
                entities.modules.add(match.split('.')[1]);
            });
        }
        
        // Extract functions
        const functionMatches = pageText.match(/def\s+(\w+)/g);
        if (functionMatches) {
            functionMatches.forEach(match => {
                entities.functions.add(match.replace('def ', ''));
            });
        }
        
        // Extract classes
        const classMatches = pageText.match(/class\s+(\w+)/g);
        if (classMatches) {
            classMatches.forEach(match => {
                entities.classes.add(match.replace('class ', ''));
            });
        }
        
        // Convert sets to arrays
        return {
            modules: Array.from(entities.modules),
            functions: Array.from(entities.functions),
            classes: Array.from(entities.classes),
            parameters: Array.from(entities.parameters)
        };
    }

    highlightPyTorchEntities() {
        // Highlight PyTorch function calls
        this.highlightPattern(/torch\.\w+\.\w+\(/g, 'pytorch-rag-function-highlight');
        
        // Highlight class references
        this.highlightPattern(/torch\.nn\.\w+/g, 'pytorch-rag-class-highlight');
        
        // Highlight code blocks
        const codeBlocks = document.querySelectorAll('pre, code, .highlight');
        codeBlocks.forEach(block => {
            if (block.textContent.includes('torch')) {
                block.classList.add('pytorch-rag-code-highlight');
            }
        });
    }

    highlightPattern(pattern, className) {
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const textNodes = [];
        let node;
        
        while (node = walker.nextNode()) {
            if (node.textContent.match(pattern)) {
                textNodes.push(node);
            }
        }
        
        textNodes.forEach(textNode => {
            const parent = textNode.parentNode;
            if (parent.tagName !== 'SCRIPT' && parent.tagName !== 'STYLE') {
                const highlightedHTML = textNode.textContent.replace(
                    pattern,
                    `<span class="${className}">$&</span>`
                );
                
                if (highlightedHTML !== textNode.textContent) {
                    const wrapper = document.createElement('span');
                    wrapper.innerHTML = highlightedHTML;
                    parent.replaceChild(wrapper, textNode);
                }
            }
        });
    }

    addCodeTooltips() {
        const codeElements = document.querySelectorAll('code, pre, .highlight');
        
        codeElements.forEach(element => {
            if (element.textContent.includes('torch')) {
                element.addEventListener('mouseenter', (e) => {
                    this.showTooltip(e.target, 'Click to ask about this code');
                });
                
                element.addEventListener('mouseleave', () => {
                    this.hideTooltip();
                });
                
                element.addEventListener('click', (e) => {
                    this.onCodeClick(e.target);
                });
            }
        });
    }

    showTooltip(element, text) {
        const tooltip = document.createElement('div');
        tooltip.className = 'pytorch-rag-tooltip';
        tooltip.textContent = text;
        
        document.body.appendChild(tooltip);
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
        
        setTimeout(() => {
            tooltip.classList.add('show');
        }, 100);
        
        this.currentTooltip = tooltip;
    }

    hideTooltip() {
        if (this.currentTooltip) {
            this.currentTooltip.remove();
            this.currentTooltip = null;
        }
    }

    onCodeClick(element) {
        const code = element.textContent.trim();
        if (code.length > 5) {
            // Send message to popup to ask about this code
            chrome.runtime.sendMessage({
                type: 'CODE_SELECTED',
                data: {
                    code: code,
                    context: this.extractPageContext()
                }
            });
        }
    }

    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            switch (request.type) {
                case 'GET_PAGE_CONTENT':
                    sendResponse(this.extractPageContext());
                    break;
                    
                case 'HIGHLIGHT_ELEMENTS':
                    this.highlightElements(request.data.elements);
                    break;
                    
                case 'CLEAR_HIGHLIGHTS':
                    this.clearHighlights();
                    break;
                    
                default:
                    sendResponse({ error: 'Unknown message type' });
            }
        });
    }

    setupPageChangeDetection() {
        // Detect page changes in SPAs
        let currentUrl = window.location.href;
        
        const observer = new MutationObserver(() => {
            if (window.location.href !== currentUrl) {
                currentUrl = window.location.href;
                setTimeout(() => {
                    this.analyzePage();
                }, 1000);
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Also listen for popstate events
        window.addEventListener('popstate', () => {
            setTimeout(() => {
                this.analyzePage();
            }, 1000);
        });
    }

    highlightElements(elements) {
        this.clearHighlights();
        
        elements.forEach(element => {
            if (element.selector) {
                const el = document.querySelector(element.selector);
                if (el) {
                    el.classList.add('pytorch-rag-highlight');
                    this.highlightedElements.push(el);
                }
            }
        });
        
        // Auto-clear highlights after 5 seconds
        setTimeout(() => {
            this.clearHighlights();
        }, 5000);
    }

    clearHighlights() {
        this.highlightedElements.forEach(element => {
            element.classList.remove('pytorch-rag-highlight');
        });
        this.highlightedElements = [];
    }

    openRAGPopup() {
        // Send message to background script to open popup
        chrome.runtime.sendMessage({
            type: 'OPEN_POPUP',
            data: {
                context: this.extractPageContext()
            }
        });
    }
}

// Initialize content script
new PyTorchContentScript();
