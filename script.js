// QuantBot AI - Ultimate Neural Trading System
// Complete Implementation with Self-Trading Simulator

class QuantBotAI {
    constructor() {
        this.isConnected = false;
        this.llmEndpoint = 'http://localhost:11434';
        this.currentModel = 'llama2';
        this.simulatorRunning = false;
        this.portfolio = {
            balance: 100000,
            positions: {},
            totalValue: 100000,
            dailyPnL: 0,
            totalReturn: 0
        };
        this.tradeHistory = [];
        this.marketData = {};
        this.charts = {};
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.startMarketDataSimulation();
        this.startRealTimeUpdates();
        
        // Auto-start simulator after 3 seconds
        setTimeout(() => {
            this.startTradingSimulator();
        }, 3000);
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = e.target.getAttribute('href').substring(1);
                this.switchTab(target);
            });
        });

        // LLM Connection
        document.getElementById('connectLLM').addEventListener('click', () => {
            document.getElementById('llmModal').style.display = 'flex';
        });

        document.getElementById('closeLLMModal').addEventListener('click', () => {
            document.getElementById('llmModal').style.display = 'none';
        });

        document.getElementById('testConnection').addEventListener('click', () => {
            this.testLLMConnection();
        });

        document.getElementById('connectButton').addEventListener('click', () => {
            this.connectToLLM();
        });

        // Chat
        document.getElementById('sendMessage').addEventListener('click', () => {
            this.sendChatMessage();
        });

        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });

        // Trading Simulator Controls
        document.addEventListener('click', (e) => {
            if (e.target.id === 'startSimulator') {
                this.startTradingSimulator();
            } else if (e.target.id === 'stopSimulator') {
                this.stopTradingSimulator();
            }
        });
    }

    switchTab(tabName) {
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[href="#${tabName}"]`).classList.add('active');

        // Hide all sections
        document.querySelectorAll('.dashboard-section, .tab-content').forEach(section => {
            section.style.display = 'none';
        });

        // Show target section or create it
        let targetSection = document.getElementById(tabName);
        
        if (!targetSection) {
            targetSection = this.createTabContent(tabName);
        }
        
        if (tabName === 'dashboard') {
            document.querySelector('.dashboard-section').style.display = 'block';
        } else {
            targetSection.style.display = 'block';
        }
    }

    createTabContent(tabName) {
        const mainContent = document.querySelector('.main-content');
        const section = document.createElement('section');
        section.id = tabName;
        section.className = 'tab-content';
        
        const content = this.getTabContent(tabName);
        section.innerHTML = content;
        
        mainContent.appendChild(section);
        return section;
    }

    getTabContent(tabName) {
        const contents = {
            strategies: `
                <div class="tab-container">
                    <div class="tab-header">
                        <h2 class="tab-title">
                            <i class="fas fa-robot neural-icon"></i>
                            Neural Strategy Engine
                        </h2>
                        <div class="tab-actions">
                            <button class="btn-neural" id="startSimulator">
                                <i class="fas fa-play"></i>
                                Start Simulator
                            </button>
                            <button class="btn-secondary" id="stopSimulator">
                                <i class="fas fa-stop"></i>
                                Stop Simulator
                            </button>
                        </div>
                    </div>
                    <div class="strategies-grid">
                        <div class="strategy-card active">
                            <div class="strategy-header">
                                <h3>Neural Momentum Strategy</h3>
                                <div class="strategy-status running">
                                    <div class="status-dot"></div>
                                    <span id="simulatorStatus">Running</span>
                                </div>
                            </div>
                            <div class="strategy-metrics">
                                <div class="metric">
                                    <span class="metric-label">Performance</span>
                                    <span class="metric-value neural-glow">+23.7%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Sharpe Ratio</span>
                                    <span class="metric-value neural-glow">2.84</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Max Drawdown</span>
                                    <span class="metric-value neural-glow">-3.2%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Win Rate</span>
                                    <span class="metric-value neural-glow">87%</span>
                                </div>
                            </div>
                        </div>
                        <div class="strategy-card">
                            <div class="strategy-header">
                                <h3>AI Sentiment Strategy</h3>
                                <div class="strategy-status standby">
                                    <div class="status-dot"></div>
                                    <span>Standby</span>
                                </div>
                            </div>
                            <div class="strategy-metrics">
                                <div class="metric">
                                    <span class="metric-label">Performance</span>
                                    <span class="metric-value neural-glow">+18.3%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Sharpe Ratio</span>
                                    <span class="metric-value neural-glow">2.12</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Max Drawdown</span>
                                    <span class="metric-value neural-glow">-4.1%</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Win Rate</span>
                                    <span class="metric-value neural-glow">82%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            portfolio: `
                <div class="tab-container">
                    <div class="tab-header">
                        <h2 class="tab-title">
                            <i class="fas fa-chart-pie neural-icon"></i>
                            Neural Portfolio Management
                        </h2>
                    </div>
                    <div class="portfolio-overview">
                        <div class="portfolio-summary-card">
                            <h3>Portfolio Summary</h3>
                            <div class="summary-metrics">
                                <div class="summary-item">
                                    <span class="summary-label">Total Value</span>
                                    <span class="summary-value neural-glow" id="portfolioTotalValue">$100,000.00</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Daily P&L</span>
                                    <span class="summary-value neural-glow" id="portfolioDailyPnL">$0.00</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Total Return</span>
                                    <span class="summary-value neural-glow" id="portfolioTotalReturn">0.00%</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Available Cash</span>
                                    <span class="summary-value neural-glow" id="portfolioCash">$100,000.00</span>
                                </div>
                            </div>
                        </div>
                        <div class="positions-card">
                            <h3>Current Positions</h3>
                            <div class="positions-list" id="positionsList">
                                <div class="no-positions">No positions yet. Simulator will start trading soon...</div>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            analytics: `
                <div class="tab-container">
                    <div class="tab-header">
                        <h2 class="tab-title">
                            <i class="fas fa-chart-line neural-icon"></i>
                            Advanced Analytics
                        </h2>
                    </div>
                    <div class="analytics-grid">
                        <div class="analytics-card">
                            <h3>Performance Metrics</h3>
                            <div class="analytics-metrics">
                                <div class="analytics-item">
                                    <span class="analytics-label">Sharpe Ratio</span>
                                    <span class="analytics-value neural-glow">2.84</span>
                                </div>
                                <div class="analytics-item">
                                    <span class="analytics-label">Max Drawdown</span>
                                    <span class="analytics-value neural-glow">-3.2%</span>
                                </div>
                                <div class="analytics-item">
                                    <span class="analytics-label">Win Rate</span>
                                    <span class="analytics-value neural-glow">87%</span>
                                </div>
                                <div class="analytics-item">
                                    <span class="analytics-label">Profit Factor</span>
                                    <span class="analytics-value neural-glow">3.47</span>
                                </div>
                            </div>
                        </div>
                        <div class="analytics-card">
                            <h3>Risk Metrics</h3>
                            <div class="analytics-metrics">
                                <div class="analytics-item">
                                    <span class="analytics-label">VaR (95%)</span>
                                    <span class="analytics-value neural-glow">-2.1%</span>
                                </div>
                                <div class="analytics-item">
                                    <span class="analytics-label">Beta</span>
                                    <span class="analytics-value neural-glow">0.73</span>
                                </div>
                                <div class="analytics-item">
                                    <span class="analytics-label">Volatility</span>
                                    <span class="analytics-value neural-glow">12.4%</span>
                                </div>
                                <div class="analytics-item">
                                    <span class="analytics-label">Correlation</span>
                                    <span class="analytics-value neural-glow">0.68</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            settings: `
                <div class="tab-container">
                    <div class="tab-header">
                        <h2 class="tab-title">
                            <i class="fas fa-cog neural-icon"></i>
                            System Configuration
                        </h2>
                    </div>
                    <div class="settings-grid">
                        <div class="settings-card">
                            <h3>Neural Engine Settings</h3>
                            <div class="settings-group">
                                <label>Processing Mode</label>
                                <select class="neural-select">
                                    <option>M3 Max Optimized</option>
                                    <option>High Performance</option>
                                    <option>Balanced</option>
                                </select>
                            </div>
                            <div class="settings-group">
                                <label>AI Model</label>
                                <select class="neural-select">
                                    <option>Llama 2 (Neural)</option>
                                    <option>Code Llama</option>
                                    <option>Mistral</option>
                                </select>
                            </div>
                        </div>
                        <div class="settings-card">
                            <h3>Trading Parameters</h3>
                            <div class="settings-group">
                                <label>Max Position Size</label>
                                <input type="range" min="1" max="20" value="10" class="neural-slider">
                                <span>10%</span>
                            </div>
                            <div class="settings-group">
                                <label>Risk Level</label>
                                <select class="neural-select">
                                    <option>Conservative</option>
                                    <option>Moderate</option>
                                    <option>Aggressive</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            neural: `
                <div class="tab-container">
                    <div class="tab-header">
                        <h2 class="tab-title">
                            <i class="fas fa-brain neural-icon"></i>
                            M3 Max Neural Control Panel
                        </h2>
                    </div>
                    <div class="neural-control-grid">
                        <div class="neural-control-card">
                            <h3>Processing Metrics</h3>
                            <div class="neural-metrics">
                                <div class="neural-metric">
                                    <span class="metric-label">Neural Cores Active</span>
                                    <span class="metric-value neural-glow">16/16</span>
                                </div>
                                <div class="neural-metric">
                                    <span class="metric-label">Processing Speed</span>
                                    <span class="metric-value neural-glow">15.8 TOPS</span>
                                </div>
                                <div class="neural-metric">
                                    <span class="metric-label">Memory Usage</span>
                                    <span class="metric-value neural-glow">47.3 GB</span>
                                </div>
                                <div class="neural-metric">
                                    <span class="metric-label">Temperature</span>
                                    <span class="metric-value neural-glow">42°C</span>
                                </div>
                            </div>
                        </div>
                        <div class="neural-control-card">
                            <h3>AI Model Status</h3>
                            <div class="model-status">
                                <div class="model-item">
                                    <span class="model-name">Llama 2 (Neural)</span>
                                    <div class="model-indicator active"></div>
                                </div>
                                <div class="model-item">
                                    <span class="model-name">Code Llama</span>
                                    <div class="model-indicator standby"></div>
                                </div>
                                <div class="model-item">
                                    <span class="model-name">Mistral</span>
                                    <div class="model-indicator standby"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `
        };
        
        return contents[tabName] || '<div class="tab-container"><h2>Coming Soon</h2></div>';
    }

    async testLLMConnection() {
        const testDiv = document.getElementById('connectionTest');
        testDiv.style.display = 'block';
        
        const endpoint = document.getElementById('llmEndpoint').value;
        
        try {
            const response = await fetch(`${endpoint}/api/tags`);
            if (response.ok) {
                testDiv.innerHTML = `
                    <div class="test-status success">
                        <i class="fas fa-check-circle"></i>
                        <span>Connection successful!</span>
                    </div>
                `;
            } else {
                throw new Error('Connection failed');
            }
        } catch (error) {
            testDiv.innerHTML = `
                <div class="test-status error">
                    <i class="fas fa-times-circle"></i>
                    <span>Connection failed. Check endpoint and try again.</span>
                </div>
            `;
        }
    }

    async connectToLLM() {
        const modal = document.getElementById('llmModal');
        const endpoint = document.getElementById('llmEndpoint').value;
        const model = document.getElementById('llmModel').value;
        
        // Create connection progress overlay
        const progressOverlay = document.createElement('div');
        progressOverlay.className = 'connection-progress-overlay';
        progressOverlay.innerHTML = `
            <div class="connection-progress">
                <div class="neural-connection-viz">
                    <div class="connection-nodes">
                        <div class="node" id="node1"></div>
                        <div class="node" id="node2"></div>
                        <div class="node" id="node3"></div>
                        <div class="node" id="node4"></div>
                        <div class="node" id="node5"></div>
                    </div>
                    <div class="connection-lines">
                        <div class="line" id="line1"></div>
                        <div class="line" id="line2"></div>
                        <div class="line" id="line3"></div>
                        <div class="line" id="line4"></div>
                    </div>
                </div>
                <div class="connection-status">
                    <h3>Establishing Neural Connection</h3>
                    <div class="status-text" id="connectionStatusText">Initializing M3 Max Neural Engine...</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="connectionProgress"></div>
                    </div>
                </div>
                <div class="connection-log" id="connectionLog">
                    <div class="log-entry">🧠 Initializing neural pathways...</div>
                </div>
            </div>
        `;
        
        modal.appendChild(progressOverlay);
        
        // Simulate connection process
        const steps = [
            { text: "🔍 Scanning for M3 Max neural cores...", progress: 20 },
            { text: "⚡ Activating neural processing units...", progress: 40 },
            { text: "🧠 Loading AI model weights...", progress: 60 },
            { text: "🔗 Establishing quantum entanglement...", progress: 80 },
            { text: "✅ Neural connection established!", progress: 100 }
        ];
        
        for (let i = 0; i < steps.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            const statusText = document.getElementById('connectionStatusText');
            const progress = document.getElementById('connectionProgress');
            const log = document.getElementById('connectionLog');
            const node = document.getElementById(`node${i + 1}`);
            const line = document.getElementById(`line${i + 1}`);
            
            if (statusText) statusText.textContent = steps[i].text;
            if (progress) progress.style.width = steps[i].progress + '%';
            if (log) {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                logEntry.textContent = steps[i].text;
                log.appendChild(logEntry);
                log.scrollTop = log.scrollHeight;
            }
            if (node) node.classList.add('active');
            if (line) line.classList.add('active');
        }
        
        // Connection successful
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        this.isConnected = true;
        this.llmEndpoint = endpoint;
        this.currentModel = model;
        
        // Update UI
        document.getElementById('llmStatus').innerHTML = `
            <div class="neural-status-dot connected"></div>
            <span>M3 Max Connected</span>
        `;
        
        document.getElementById('chatInput').disabled = false;
        document.getElementById('sendMessage').disabled = false;
        
        // Close modal
        modal.style.display = 'none';
        modal.removeChild(progressOverlay);
        
        // Add success message to chat
        this.addChatMessage('assistant', '🧠 Neural connection established! M3 Max is now online and ready for quantum-level market analysis.');
    }

    async sendChatMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message || !this.isConnected) return;
        
        // Add user message
        this.addChatMessage('user', message);
        input.value = '';
        
        // Add typing indicator
        const typingId = this.addChatMessage('assistant', '🧠 Neural processing...', true);
        
        try {
            // Simulate AI response (replace with actual LLM call)
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Remove typing indicator
            document.getElementById(typingId).remove();
            
            // Generate AI response based on message content
            const response = this.generateAIResponse(message);
            this.addChatMessage('assistant', response);
            
        } catch (error) {
            document.getElementById(typingId).remove();
            this.addChatMessage('assistant', '❌ Neural processing error. Please try again.');
        }
    }

    generateAIResponse(message) {
        const responses = {
            portfolio: "📊 Your neural portfolio is performing exceptionally well with a 23.7% return and 2.84 Sharpe ratio. The AI has identified optimal entry points in NVDA and TSLA based on momentum patterns.",
            market: "📈 Current market analysis shows strong bullish momentum in tech stocks. Neural sentiment analysis indicates 94% positive sentiment with high confidence levels.",
            risk: "🛡️ Risk metrics are within optimal parameters. VaR at -2.1% with minimal correlation exposure. The neural risk shield is actively protecting your positions.",
            strategy: "🤖 The momentum strategy is outperforming with 87% win rate. AI recommends maintaining current positions while monitoring for breakout patterns.",
            default: "🧠 Neural analysis complete. The M3 Max is processing market data at 15.8 TOPS with 98.7% accuracy. How can I assist with your trading strategy?"
        };
        
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('portfolio') || lowerMessage.includes('performance')) {
            return responses.portfolio;
        } else if (lowerMessage.includes('market') || lowerMessage.includes('analysis')) {
            return responses.market;
        } else if (lowerMessage.includes('risk') || lowerMessage.includes('drawdown')) {
            return responses.risk;
        } else if (lowerMessage.includes('strategy') || lowerMessage.includes('trading')) {
            return responses.strategy;
        } else {
            return responses.default;
        }
    }

    addChatMessage(sender, message, isTyping = false) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        const messageId = 'msg_' + Date.now();
        
        messageDiv.id = messageId;
        messageDiv.className = `message ${sender}`;
        
        if (sender === 'assistant') {
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-brain neural-icon"></i>
                </div>
                <div class="message-content">
                    <p>${message}</p>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <p>${message}</p>
                </div>
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
            `;
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageId;
    }

    startTradingSimulator() {
        if (this.simulatorRunning) return;
        
        this.simulatorRunning = true;
        
        // Update UI
        const statusElement = document.getElementById('simulatorStatus');
        if (statusElement) {
            statusElement.textContent = 'Running';
            statusElement.parentElement.className = 'strategy-status running';
        }
        
        // Start trading loop
        this.tradingInterval = setInterval(() => {
            this.executeNeuralTrade();
        }, 5000); // Trade every 5 seconds
        
        console.log('🤖 Neural Trading Simulator Started');
    }

    stopTradingSimulator() {
        if (!this.simulatorRunning) return;
        
        this.simulatorRunning = false;
        
        if (this.tradingInterval) {
            clearInterval(this.tradingInterval);
        }
        
        // Update UI
        const statusElement = document.getElementById('simulatorStatus');
        if (statusElement) {
            statusElement.textContent = 'Stopped';
            statusElement.parentElement.className = 'strategy-status stopped';
        }
        
        console.log('🛑 Neural Trading Simulator Stopped');
    }

    executeNeuralTrade() {
        const symbols = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN'];
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        
        // Get current market data
        const currentPrice = this.marketData[symbol]?.price || this.generatePrice(symbol);
        
        // Neural AI decision making
        const analysis = this.performNeuralAnalysis(symbol, currentPrice);
        
        if (analysis.confidence > 0.7 && analysis.signal !== 'hold') {
            const trade = this.executeTrade(symbol, analysis.signal, currentPrice, analysis.confidence);
            if (trade) {
                this.addTradeToHistory(trade);
                this.updatePortfolioDisplay();
            }
        }
    }

    performNeuralAnalysis(symbol, price) {
        // Simulate neural analysis combining multiple factors
        const technicalScore = Math.random() * 2 - 1; // -1 to 1
        const sentimentScore = Math.random() * 2 - 1;
        const riskScore = Math.random() * 2 - 1;
        
        // Combine scores with weights
        const combinedScore = (technicalScore * 0.4) + (sentimentScore * 0.3) + (riskScore * 0.3);
        const confidence = Math.abs(combinedScore);
        
        let signal = 'hold';
        if (combinedScore > 0.3) signal = 'buy';
        else if (combinedScore < -0.3) signal = 'sell';
        
        return {
            signal,
            confidence,
            technicalScore,
            sentimentScore,
            riskScore
        };
    }

    executeTrade(symbol, action, price, confidence) {
        const maxPositionValue = this.portfolio.totalValue * 0.1; // Max 10% per position
        const tradeValue = Math.min(maxPositionValue, confidence * 10000); // Scale by confidence
        const quantity = Math.floor(tradeValue / price);
        
        if (quantity === 0) return null;
        
        const trade = {
            id: Date.now(),
            timestamp: new Date(),
            symbol,
            action,
            quantity,
            price,
            value: quantity * price,
            confidence: Math.round(confidence * 100)
        };
        
        // Update portfolio
        if (action === 'buy') {
            if (this.portfolio.balance >= trade.value) {
                this.portfolio.balance -= trade.value;
                this.portfolio.positions[symbol] = this.portfolio.positions[symbol] || { quantity: 0, avgPrice: 0 };
                
                const currentPos = this.portfolio.positions[symbol];
                const totalQuantity = currentPos.quantity + quantity;
                const totalValue = (currentPos.quantity * currentPos.avgPrice) + trade.value;
                
                this.portfolio.positions[symbol] = {
                    quantity: totalQuantity,
                    avgPrice: totalValue / totalQuantity
                };
            } else {
                return null; // Insufficient funds
            }
        } else if (action === 'sell' && this.portfolio.positions[symbol]?.quantity >= quantity) {
            this.portfolio.balance += trade.value;
            this.portfolio.positions[symbol].quantity -= quantity;
            
            if (this.portfolio.positions[symbol].quantity === 0) {
                delete this.portfolio.positions[symbol];
            }
        } else {
            return null; // Cannot sell what we don't have
        }
        
        this.updatePortfolioValue();
        return trade;
    }

    addTradeToHistory(trade) {
        this.tradeHistory.unshift(trade);
        
        // Keep only last 50 trades
        if (this.tradeHistory.length > 50) {
            this.tradeHistory = this.tradeHistory.slice(0, 50);
        }
        
        // Update trades display
        this.updateTradesDisplay();
    }

    updateTradesDisplay() {
        const tradesContainer = document.querySelector('.trades-list');
        if (!tradesContainer) return;
        
        // Clear existing trades
        tradesContainer.innerHTML = '';
        
        // Add recent trades
        this.tradeHistory.slice(0, 5).forEach(trade => {
            const tradeElement = document.createElement('div');
            tradeElement.className = 'trade-item';
            tradeElement.innerHTML = `
                <div class="trade-time neural-glow">${trade.timestamp.toLocaleTimeString()}</div>
                <div class="trade-symbol neural-glow">${trade.symbol}</div>
                <div class="trade-action ${trade.action} neural-action">AI ${trade.action.toUpperCase()}</div>
                <div class="trade-quantity">${trade.quantity} shares</div>
                <div class="trade-price neural-glow">$${trade.price.toFixed(2)}</div>
                <div class="trade-status success neural-pulse">EXECUTED</div>
                <div class="trade-ai-score">AI: ${trade.confidence}%</div>
            `;
            tradesContainer.appendChild(tradeElement);
        });
    }

    updatePortfolioValue() {
        let totalValue = this.portfolio.balance;
        
        // Add value of all positions
        Object.entries(this.portfolio.positions).forEach(([symbol, position]) => {
            const currentPrice = this.marketData[symbol]?.price || position.avgPrice;
            totalValue += position.quantity * currentPrice;
        });
        
        this.portfolio.dailyPnL = totalValue - this.portfolio.totalValue;
        this.portfolio.totalValue = totalValue;
        this.portfolio.totalReturn = ((totalValue - 100000) / 100000) * 100;
    }

    updatePortfolioDisplay() {
        // Update main portfolio stats
        document.getElementById('totalReturn').textContent = `+${this.portfolio.totalReturn.toFixed(1)}%`;
        
        // Update portfolio tab if it exists
        const portfolioValue = document.getElementById('portfolioTotalValue');
        const dailyPnL = document.getElementById('portfolioDailyPnL');
        const totalReturn = document.getElementById('portfolioTotalReturn');
        const cash = document.getElementById('portfolioCash');
        
        if (portfolioValue) portfolioValue.textContent = `$${this.portfolio.totalValue.toLocaleString()}`;
        if (dailyPnL) {
            dailyPnL.textContent = `${this.portfolio.dailyPnL >= 0 ? '+' : ''}$${this.portfolio.dailyPnL.toLocaleString()}`;
            dailyPnL.className = `summary-value neural-glow ${this.portfolio.dailyPnL >= 0 ? 'positive' : 'negative'}`;
        }
        if (totalReturn) {
            totalReturn.textContent = `${this.portfolio.totalReturn >= 0 ? '+' : ''}${this.portfolio.totalReturn.toFixed(2)}%`;
            totalReturn.className = `summary-value neural-glow ${this.portfolio.totalReturn >= 0 ? 'positive' : 'negative'}`;
        }
        if (cash) cash.textContent = `$${this.portfolio.balance.toLocaleString()}`;
        
        // Update positions list
        this.updatePositionsList();
    }

    updatePositionsList() {
        const positionsList = document.getElementById('positionsList');
        if (!positionsList) return;
        
        if (Object.keys(this.portfolio.positions).length === 0) {
            positionsList.innerHTML = '<div class="no-positions">No positions yet. Simulator will start trading soon...</div>';
            return;
        }
        
        positionsList.innerHTML = '';
        
        Object.entries(this.portfolio.positions).forEach(([symbol, position]) => {
            const currentPrice = this.marketData[symbol]?.price || position.avgPrice;
            const marketValue = position.quantity * currentPrice;
            const pnl = marketValue - (position.quantity * position.avgPrice);
            const pnlPercent = (pnl / (position.quantity * position.avgPrice)) * 100;
            
            const positionElement = document.createElement('div');
            positionElement.className = 'position-item';
            positionElement.innerHTML = `
                <div class="position-symbol neural-glow">${symbol}</div>
                <div class="position-details">
                    <span class="position-shares">${position.quantity} shares</span>
                    <span class="position-value">$${marketValue.toLocaleString()}</span>
                </div>
                <div class="position-pnl ${pnl >= 0 ? 'positive' : 'negative'} neural-pulse">
                    ${pnl >= 0 ? '+' : ''}${pnlPercent.toFixed(1)}%
                </div>
                <div class="position-price">
                    <span>Avg: $${position.avgPrice.toFixed(2)}</span>
                    <span>Current: $${currentPrice.toFixed(2)}</span>
                </div>
            `;
            positionsList.appendChild(positionElement);
        });
    }

    generatePrice(symbol) {
        const basePrices = {
            'NVDA': 850,
            'TSLA': 250,
            'AAPL': 190,
            'GOOGL': 140,
            'MSFT': 420,
            'AMZN': 150
        };
        
        const basePrice = basePrices[symbol] || 100;
        const variation = (Math.random() - 0.5) * 0.02; // ±1% variation
        return basePrice * (1 + variation);
    }

    startMarketDataSimulation() {
        const symbols = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN'];
        
        // Initialize market data
        symbols.forEach(symbol => {
            this.marketData[symbol] = {
                price: this.generatePrice(symbol),
                change: 0,
                changePercent: 0
            };
        });
        
        // Update market data every 2 seconds
        setInterval(() => {
            symbols.forEach(symbol => {
                const oldPrice = this.marketData[symbol].price;
                const newPrice = this.generatePrice(symbol);
                const change = newPrice - oldPrice;
                const changePercent = (change / oldPrice) * 100;
                
                this.marketData[symbol] = {
                    price: newPrice,
                    change,
                    changePercent
                };
            });
            
            this.updateMarketDataDisplay();
            this.updatePortfolioValue();
            this.updatePortfolioDisplay();
        }, 2000);
    }

    updateMarketDataDisplay() {
        // Update equity list
        const equitiesList = document.getElementById('equitiesList');
        if (equitiesList) {
            equitiesList.innerHTML = '';
            Object.entries(this.marketData).slice(0, 3).forEach(([symbol, data]) => {
                const assetElement = document.createElement('div');
                assetElement.className = 'asset-item';
                assetElement.innerHTML = `
                    <span class="symbol">${symbol}</span>
                    <span class="price neural-glow">$${data.price.toFixed(2)}</span>
                    <span class="change ${data.changePercent >= 0 ? 'positive' : 'negative'}">
                        ${data.changePercent >= 0 ? '+' : ''}${data.changePercent.toFixed(2)}%
                    </span>
                `;
                equitiesList.appendChild(assetElement);
            });
        }
    }

    initializeCharts() {
        // Initialize hero chart
        const heroCanvas = document.getElementById('heroChart');
        if (heroCanvas) {
            const ctx = heroCanvas.getContext('2d');
            this.charts.hero = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 50}, (_, i) => i),
                    datasets: [{
                        label: 'Portfolio Value',
                        data: Array.from({length: 50}, () => Math.random() * 1000 + 100000),
                        borderColor: '#00ffff',
                        backgroundColor: 'rgba(0, 255, 255, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { display: false },
                        y: { display: false }
                    },
                    elements: {
                        point: { radius: 0 }
                    }
                }
            });
        }
    }

    startRealTimeUpdates() {
        // Update neural metrics every 3 seconds
        setInterval(() => {
            this.updateNeuralMetrics();
        }, 3000);
        
        // Update news feed every 10 seconds
        setInterval(() => {
            this.updateNewsFeed();
        }, 10000);
    }

    updateNeuralMetrics() {
        // Update processing speed
        const processingSpeed = document.getElementById('processingSpeed');
        if (processingSpeed) {
            const speed = (15.5 + Math.random() * 0.6).toFixed(1);
            processingSpeed.textContent = `${speed} TOPS + Live`;
        }
        
        // Update model accuracy
        const modelAccuracy = document.getElementById('modelAccuracy');
        if (modelAccuracy) {
            const accuracy = (98.5 + Math.random() * 0.4).toFixed(1);
            modelAccuracy.textContent = `${accuracy}%`;
        }
        
        // Update predictions per second
        const predictionsPerSec = document.getElementById('optionsChains');
        if (predictionsPerSec) {
            const predictions = Math.floor(2800 + Math.random() * 100);
            predictionsPerSec.textContent = predictions.toLocaleString();
        }
    }

    updateNewsFeed() {
        const newsFeed = document.getElementById('newsFeed');
        if (!newsFeed) return;
        
        const newsItems = [
            {
                source: 'Reuters',
                headline: 'NVIDIA Reports Record Q4 Earnings, Beats Estimates',
                sentiment: 'Bullish 94%',
                symbol: 'NVDA',
                prediction: '+3.2%'
            },
            {
                source: 'Bloomberg',
                headline: 'Tesla Announces New Gigafactory Expansion',
                sentiment: 'Bullish 87%',
                symbol: 'TSLA',
                prediction: '+2.1%'
            },
            {
                source: 'CNBC',
                headline: 'Apple Unveils M4 Max Chip with Enhanced Neural Engine',
                sentiment: 'Bullish 91%',
                symbol: 'AAPL',
                prediction: '+1.8%'
            }
        ];
        
        // Add new news item occasionally
        if (Math.random() < 0.3) {
            const randomNews = newsItems[Math.floor(Math.random() * newsItems.length)];
            const newsElement = document.createElement('div');
            newsElement.className = 'news-item';
            newsElement.innerHTML = `
                <div class="news-time">${new Date().toLocaleTimeString()}</div>
                <div class="news-source">${randomNews.source}</div>
                <div class="news-headline">${randomNews.headline}</div>
                <div class="news-sentiment positive neural-glow">${randomNews.sentiment}</div>
                <div class="news-impact">
                    <span class="impact-symbol">${randomNews.symbol}</span>
                    <span class="impact-prediction">${randomNews.prediction}</span>
                </div>
            `;
            
            newsFeed.insertBefore(newsElement, newsFeed.firstChild);
            
            // Keep only last 10 news items
            while (newsFeed.children.length > 10) {
                newsFeed.removeChild(newsFeed.lastChild);
            }
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Hide loading screen
    setTimeout(() => {
        document.getElementById('loadingScreen').style.display = 'none';
    }, 2000);
    
    // Initialize QuantBot AI
    window.quantBot = new QuantBotAI();
});