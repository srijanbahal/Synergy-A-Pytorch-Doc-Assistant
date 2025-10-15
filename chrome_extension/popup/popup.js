/**
 * PyTorch RAG Assistant - Popup Script
 */

class PyTorchRAGPopup {
    constructor() {
        this.apiEndpoint = 'http://localhost:8000';
        this.sessionId = null;
        this.chatHistory = [];
        this.settings = {
            apiEndpoint: 'http://localhost:8000',
            model: 'llama3.1:8b',
            maxHistory: 10,
            autoScroll: true,
            showCitations: true
        };
        
        this.initializeElements();
        this.loadSettings();
        this.setupEventListeners();
        this.checkConnection();
        this.generateSessionId();
    }

    initializeElements() {
        // Main elements
        this.chatMessages = document.getElementById('chatMessages');
        this.queryInput = document.getElementById('queryInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        
        // Header buttons
        this.settingsBtn = document.getElementById('settingsBtn');
        this.clearBtn = document.getElementById('clearBtn');
        
        // Input elements
        this.attachBtn = document.getElementById('attachBtn');
        this.includePageContext = document.getElementById('includePageContext');
        this.charCount = document.getElementById('charCount');
        
        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.statusText = document.getElementById('statusText');
        this.sessionInfo = document.getElementById('sessionInfo');
        
        // Modals
        this.settingsModal = document.getElementById('settingsModal');
        this.citationModal = document.getElementById('citationModal');
        
        // Settings elements
        this.apiEndpointInput = document.getElementById('apiEndpoint');
        this.modelSelect = document.getElementById('modelSelect');
        this.maxHistoryInput = document.getElementById('maxHistory');
        this.autoScrollCheckbox = document.getElementById('autoScroll');
        this.showCitationsCheckbox = document.getElementById('showCitations');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');
        this.resetSettingsBtn = document.getElementById('resetSettingsBtn');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        
        // Citation elements
        this.citationList = document.getElementById('citationList');
        this.closeCitationBtn = document.getElementById('closeCitationBtn');
    }

    setupEventListeners() {
        // Send button
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter key to send
        this.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Character count
        this.queryInput.addEventListener('input', () => {
            const count = this.queryInput.value.length;
            this.charCount.textContent = `${count}/1000`;
            this.sendBtn.disabled = count === 0;
            
            // Auto-resize textarea
            this.queryInput.style.height = 'auto';
            this.queryInput.style.height = Math.min(this.queryInput.scrollHeight, 120) + 'px';
        });
        
        // Header buttons
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.clearBtn.addEventListener('click', () => this.clearChat());
        
        // Settings modal
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.resetSettingsBtn.addEventListener('click', () => this.resetSettings());
        
        // Citation modal
        this.closeCitationBtn.addEventListener('click', () => this.closeCitations());
        
        // Click outside modal to close
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettings();
            }
        });
        
        this.citationModal.addEventListener('click', (e) => {
            if (e.target === this.citationModal) {
                this.closeCitations();
            }
        });
    }

    async sendMessage() {
        const question = this.queryInput.value.trim();
        if (!question) return;
        
        // Clear input
        this.queryInput.value = '';
        this.queryInput.style.height = 'auto';
        this.charCount.textContent = '0/1000';
        this.sendBtn.disabled = true;
        
        // Add user message to chat
        this.addMessage('user', question);
        
        // Show loading indicator
        this.showLoading(true);
        
        try {
            // Get page content if enabled
            let pageContent = null;
            let pageUrl = null;
            
            if (this.includePageContext.checked) {
                const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
                if (tab && tab.url && tab.url.includes('pytorch.org')) {
                    pageUrl = tab.url;
                    pageContent = await this.getPageContent(tab.id);
                }
            }
            
            // Send request to API
            const response = await this.callAPI('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    page_url: pageUrl,
                    page_content: pageContent,
                    session_id: this.sessionId,
                    chat_history: this.chatHistory.slice(-this.settings.maxHistory),
                    include_citations: this.settings.showCitations
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add assistant response to chat
            this.addMessage('assistant', data.answer, {
                citations: data.citations,
                confidence: data.confidence,
                metadata: data.pipeline_metrics
            });
            
            // Update session info
            this.sessionInfo.textContent = `Session: ${this.sessionId.substring(0, 8)}...`;
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
            this.updateConnectionStatus(false);
        } finally {
            this.showLoading(false);
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

    addMessage(type, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        let messageHTML = `
            <div class="message-header">
                <span class="message-type">${type === 'user' ? 'You' : 'Assistant'}</span>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-content">${this.formatMessageContent(content)}</div>
        `;
        
        // Add citations if available
        if (metadata.citations && metadata.citations.length > 0) {
            messageHTML += this.renderCitations(metadata.citations);
        }
        
        // Add confidence indicator
        if (metadata.confidence !== undefined) {
            messageHTML += this.renderConfidenceIndicator(metadata.confidence);
        }
        
        messageDiv.innerHTML = messageHTML;
        
        // Remove welcome message if it exists
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        this.chatMessages.appendChild(messageDiv);
        
        // Auto-scroll if enabled
        if (this.settings.autoScroll) {
            messageDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Update chat history
        this.chatHistory.push({
            type: type,
            content: content,
            timestamp: Date.now(),
            metadata: metadata
        });
        
        // Limit chat history
        if (this.chatHistory.length > this.settings.maxHistory * 2) {
            this.chatHistory = this.chatHistory.slice(-this.settings.maxHistory * 2);
        }
    }

    formatMessageContent(content) {
        // Escape HTML
        let formatted = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // Format code blocks
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Format inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Format links
        formatted = formatted.replace(
            /\[([^\]]+)\]\(([^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
        );
        
        // Format line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Highlight PyTorch code
        formatted = formatted.replace(
            /(torch\.\w+(?:\.\w+)*)/g,
            '<code style="color: #ee4c2c;">$1</code>'
        );
        
        return formatted;
    }

    renderCitations(citations) {
        if (!citations || citations.length === 0) return '';
        
        let citationsHTML = `
            <div class="citations">
                <div class="citations-header">Sources:</div>
        `;
        
        citations.forEach((citation, index) => {
            citationsHTML += `
                <div class="citation-item">
                    <span class="citation-number">${citation.id}.</span>
                    <a href="${citation.url}" target="_blank" class="citation-link" title="${citation.title}">
                        ${citation.title || 'Documentation'}
                    </a>
                </div>
            `;
        });
        
        citationsHTML += '</div>';
        return citationsHTML;
    }

    renderConfidenceIndicator(confidence) {
        const percentage = Math.round(confidence * 100);
        const color = confidence >= 0.8 ? '#10b981' : confidence >= 0.6 ? '#f59e0b' : '#ef4444';
        
        return `
            <div class="confidence-indicator">
                <span>Confidence: ${percentage}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${percentage}%; background-color: ${color};"></div>
                </div>
            </div>
        `;
    }

    showLoading(show) {
        this.loadingIndicator.style.display = show ? 'flex' : 'none';
        this.sendBtn.disabled = show;
    }

    async callAPI(endpoint, options = {}) {
        const url = `${this.apiEndpoint}${endpoint}`;
        return fetch(url, options);
    }

    async checkConnection() {
        try {
            const response = await this.callAPI('/api/health');
            const isConnected = response.ok;
            this.updateConnectionStatus(isConnected);
        } catch (error) {
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(isConnected) {
        if (isConnected) {
            this.connectionStatus.className = 'status-indicator online';
            this.statusText.textContent = 'Connected';
        } else {
            this.connectionStatus.className = 'status-indicator offline';
            this.statusText.textContent = 'Disconnected';
        }
    }

    generateSessionId() {
        this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🤖</div>
                <h3>Welcome to PyTorch RAG Assistant!</h3>
                <p>Ask me anything about PyTorch documentation. I can help with:</p>
                <ul>
                    <li>Function explanations and usage</li>
                    <li>Code examples and best practices</li>
                    <li>Troubleshooting and debugging</li>
                    <li>Concept explanations and tutorials</li>
                </ul>
            </div>
        `;
        
        this.chatHistory = [];
        this.generateSessionId();
        this.sessionInfo.textContent = 'Session: New';
    }

    openSettings() {
        this.settingsModal.style.display = 'flex';
        
        // Populate settings
        this.apiEndpointInput.value = this.settings.apiEndpoint;
        this.modelSelect.value = this.settings.model;
        this.maxHistoryInput.value = this.settings.maxHistory;
        this.autoScrollCheckbox.checked = this.settings.autoScroll;
        this.showCitationsCheckbox.checked = this.settings.showCitations;
    }

    closeSettings() {
        this.settingsModal.style.display = 'none';
    }

    saveSettings() {
        this.settings = {
            apiEndpoint: this.apiEndpointInput.value,
            model: this.modelSelect.value,
            maxHistory: parseInt(this.maxHistoryInput.value),
            autoScroll: this.autoScrollCheckbox.checked,
            showCitations: this.showCitationsCheckbox.checked
        };
        
        this.apiEndpoint = this.settings.apiEndpoint;
        
        // Save to storage
        chrome.storage.local.set({ settings: this.settings });
        
        // Recheck connection with new endpoint
        this.checkConnection();
        
        this.closeSettings();
    }

    resetSettings() {
        this.settings = {
            apiEndpoint: 'http://localhost:8000',
            model: 'llama3.1:8b',
            maxHistory: 10,
            autoScroll: true,
            showCitations: true
        };
        
        // Update UI
        this.apiEndpointInput.value = this.settings.apiEndpoint;
        this.modelSelect.value = this.settings.model;
        this.maxHistoryInput.value = this.settings.maxHistory;
        this.autoScrollCheckbox.checked = this.settings.autoScroll;
        this.showCitationsCheckbox.checked = this.settings.showCitations;
        
        // Save to storage
        chrome.storage.local.set({ settings: this.settings });
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.local.get(['settings']);
            if (result.settings) {
                this.settings = { ...this.settings, ...result.settings };
                this.apiEndpoint = this.settings.apiEndpoint;
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    openCitations(citations) {
        this.citationList.innerHTML = '';
        
        if (citations && citations.length > 0) {
            citations.forEach(citation => {
                const citationDiv = document.createElement('div');
                citationDiv.className = 'citation-item';
                citationDiv.innerHTML = `
                    <span class="citation-number">${citation.id}.</span>
                    <a href="${citation.url}" target="_blank" class="citation-link">
                        ${citation.title || 'Documentation'}
                    </a>
                    <div class="citation-details">
                        ${citation.module ? `Module: ${citation.module}` : ''}
                        ${citation.function ? `Function: ${citation.function}` : ''}
                    </div>
                `;
                this.citationList.appendChild(citationDiv);
            });
        }
        
        this.citationModal.style.display = 'flex';
    }

    closeCitations() {
        this.citationModal.style.display = 'none';
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PyTorchRAGPopup();
});
