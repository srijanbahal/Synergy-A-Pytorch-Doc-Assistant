/**
 * PyTorch RAG Assistant - Service Worker (Background Script)
 */

class PyTorchRAGServiceWorker {
    constructor() {
        this.apiEndpoint = 'http://localhost:8000';
        this.cache = new Map();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Handle extension installation
        chrome.runtime.onInstalled.addListener((details) => {
            this.handleInstallation(details);
        });

        // Handle messages from content scripts and popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });

        // Handle tab updates
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });

        // Handle tab activation
        chrome.tabs.onActivated.addListener((activeInfo) => {
            this.handleTabActivation(activeInfo);
        });
    }

    handleInstallation(details) {
        console.log('PyTorch RAG Assistant installed:', details.reason);
        
        // Set default settings
        chrome.storage.local.set({
            settings: {
                apiEndpoint: 'http://localhost:8000',
                model: 'llama3.1:8b',
                maxHistory: 10,
                autoScroll: true,
                showCitations: true
            }
        });

        // Create context menu items
        this.createContextMenus();
    }

    createContextMenus() {
        // Remove existing menus
        chrome.contextMenus.removeAll(() => {
            // Add main menu
            chrome.contextMenus.create({
                id: 'pytorch-rag-main',
                title: 'Ask PyTorch RAG Assistant',
                contexts: ['selection', 'page'],
                documentUrlPatterns: ['https://pytorch.org/*']
            });

            // Add submenu for selected text
            chrome.contextMenus.create({
                id: 'pytorch-rag-explain',
                parentId: 'pytorch-rag-main',
                title: 'Explain this code',
                contexts: ['selection'],
                documentUrlPatterns: ['https://pytorch.org/*']
            });

            chrome.contextMenus.create({
                id: 'pytorch-rag-examples',
                parentId: 'pytorch-rag-main',
                title: 'Show examples',
                contexts: ['selection'],
                documentUrlPatterns: ['https://pytorch.org/*']
            });

            chrome.contextMenus.create({
                id: 'pytorch-rag-debug',
                parentId: 'pytorch-rag-main',
                title: 'Help debug this',
                contexts: ['selection'],
                documentUrlPatterns: ['https://pytorch.org/*']
            });

            // Add page-level menu
            chrome.contextMenus.create({
                id: 'pytorch-rag-page-help',
                parentId: 'pytorch-rag-main',
                title: 'Help with this page',
                contexts: ['page'],
                documentUrlPatterns: ['https://pytorch.org/*']
            });
        });
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.type) {
                case 'PAGE_ANALYZED':
                    await this.handlePageAnalyzed(request.data, sender);
                    sendResponse({ success: true });
                    break;

                case 'CODE_SELECTED':
                    await this.handleCodeSelected(request.data, sender);
                    sendResponse({ success: true });
                    break;

                case 'OPEN_POPUP':
                    await this.handleOpenPopup(request.data, sender);
                    sendResponse({ success: true });
                    break;

                case 'GET_PAGE_CONTENT':
                    const content = await this.getPageContent(sender.tab.id);
                    sendResponse({ content });
                    break;

                case 'CACHE_QUERY':
                    const cachedResult = this.getCachedResult(request.data.query);
                    sendResponse({ cached: cachedResult });
                    break;

                case 'CACHE_RESULT':
                    this.cacheResult(request.data.query, request.data.result);
                    sendResponse({ success: true });
                    break;

                case 'CLEAR_CACHE':
                    this.clearCache();
                    sendResponse({ success: true });
                    break;

                default:
                    sendResponse({ error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Error handling message:', error);
            sendResponse({ error: error.message });
        }
    }

    async handlePageAnalyzed(pageContext, sender) {
        // Store page context for later use
        await chrome.storage.local.set({
            [`page_context_${sender.tab.id}`]: pageContext
        });

        // Update badge to show page is ready
        chrome.action.setBadgeText({
            text: '✓',
            tabId: sender.tab.id
        });
        chrome.action.setBadgeBackgroundColor({
            color: '#22c55e',
            tabId: sender.tab.id
        });
    }

    async handleCodeSelected(data, sender) {
        // Open popup with pre-filled query about the selected code
        await chrome.action.openPopup();
        
        // Send code context to popup
        setTimeout(() => {
            chrome.runtime.sendMessage({
                type: 'CODE_CONTEXT',
                data: {
                    code: data.code,
                    context: data.context,
                    suggestedQuery: `Can you explain this PyTorch code: ${data.code.substring(0, 100)}...`
                }
            });
        }, 100);
    }

    async handleOpenPopup(data, sender) {
        // Open popup
        await chrome.action.openPopup();
        
        // Send page context to popup
        setTimeout(() => {
            chrome.runtime.sendMessage({
                type: 'PAGE_CONTEXT',
                data: data.context
            });
        }, 100);
    }

    async handleTabUpdate(tabId, changeInfo, tab) {
        if (changeInfo.status === 'complete' && tab.url && tab.url.includes('pytorch.org')) {
            // Clear previous page context
            await chrome.storage.local.remove(`page_context_${tabId}`);
            
            // Reset badge
            chrome.action.setBadgeText({
                text: '',
                tabId: tabId
            });
        }
    }

    async handleTabActivation(activeInfo) {
        const tab = await chrome.tabs.get(activeInfo.tabId);
        
        if (tab.url && tab.url.includes('pytorch.org')) {
            // Update badge based on whether page context is available
            const result = await chrome.storage.local.get(`page_context_${activeInfo.tabId}`);
            
            if (result[`page_context_${activeInfo.tabId}`]) {
                chrome.action.setBadgeText({
                    text: '✓',
                    tabId: activeInfo.tabId
                });
                chrome.action.setBadgeBackgroundColor({
                    color: '#22c55e',
                    tabId: activeInfo.tabId
                });
            } else {
                chrome.action.setBadgeText({
                    text: '',
                    tabId: activeInfo.tabId
                });
            }
        } else {
            chrome.action.setBadgeText({
                text: '',
                tabId: activeInfo.tabId
            });
        }
    }

    async getPageContent(tabId) {
        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId: tabId },
                func: () => {
                    return {
                        url: window.location.href,
                        title: document.title,
                        content: document.body.innerText,
                        html: document.body.innerHTML
                    };
                }
            });
            
            return results[0]?.result || null;
        } catch (error) {
            console.error('Error getting page content:', error);
            return null;
        }
    }

    getCachedResult(query) {
        const cacheKey = this.generateCacheKey(query);
        const cached = this.cache.get(cacheKey);
        
        if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes
            return cached.result;
        }
        
        return null;
    }

    cacheResult(query, result) {
        const cacheKey = this.generateCacheKey(query);
        this.cache.set(cacheKey, {
            result: result,
            timestamp: Date.now()
        });
        
        // Limit cache size
        if (this.cache.size > 100) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }

    generateCacheKey(query) {
        return btoa(query.toLowerCase().trim());
    }

    clearCache() {
        this.cache.clear();
    }

    // Context menu click handler
    async handleContextMenuClick(info, tab) {
        try {
            let query = '';
            
            switch (info.menuItemId) {
                case 'pytorch-rag-explain':
                    query = `Explain this PyTorch code: ${info.selectionText}`;
                    break;
                    
                case 'pytorch-rag-examples':
                    query = `Show examples of: ${info.selectionText}`;
                    break;
                    
                case 'pytorch-rag-debug':
                    query = `Help debug this PyTorch code: ${info.selectionText}`;
                    break;
                    
                case 'pytorch-rag-page-help':
                    query = `Help me understand this PyTorch documentation page`;
                    break;
                    
                default:
                    query = info.selectionText || 'Help me with PyTorch';
            }
            
            // Open popup with pre-filled query
            await chrome.action.openPopup();
            
            // Send query to popup
            setTimeout(() => {
                chrome.runtime.sendMessage({
                    type: 'PREFILL_QUERY',
                    data: {
                        query: query,
                        context: info.selectionText || 'page'
                    }
                });
            }, 100);
            
        } catch (error) {
            console.error('Error handling context menu click:', error);
        }
    }
}

// Initialize service worker
const serviceWorker = new PyTorchRAGServiceWorker();

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    serviceWorker.handleContextMenuClick(info, tab);
});

// Handle extension startup
chrome.runtime.onStartup.addListener(() => {
    console.log('PyTorch RAG Assistant service worker started');
});

// Keep service worker alive
chrome.alarms.create('keep-alive', { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'keep-alive') {
        // Do nothing, just keep the service worker alive
    }
});
