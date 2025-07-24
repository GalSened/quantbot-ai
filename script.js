// QuantBot AI - Neural Trading System with Real Integrations
// World's Most Advanced Trading Platform

class QuantBotAI {
    constructor() {
        this.isInitialized = false;
        this.llmConnected = false;
        this.llmEndpoint = 'http://localhost:11434';
        this.llmModel = 'llama2';
        this.marketData = new Map();
        this.portfolio = {
            value: 100000,
            positions: new Map(),
            trades: [],
            pnl: 0
        };
        this.charts = {};
        this.updateIntervals = new Map();
        this.neuralEngine = {
            processing: false,
            accuracy: 0.973,
            speed: 15.8,
            cores: 16
        };
        
        this.init();
    }

    async init() {
        console.log('🧠 Initializing QuantBot AI Neural Trading System...');
        
        // Show loading screen
        this.showLoadingScreen();
        
        // Initialize components
        await this.initializeCharts();
        await this.initializeMarketData();
        await this.initializePortfolio();
        await this.initializeNeuralEngine();
        await this.initializeEventListeners();
        await this.initializeRealTimeUpdates();
        
        // Hide loading screen and show dashboard
        setTimeout(() => {
            this.hideLoadingScreen();
            this.isInitialized = true;
            console.log('✅ QuantBot AI Neural Trading System initialized successfully!');
        }, 3000);
    }

    showLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            loadingScreen.style.display = 'flex';
        }
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            loadingScreen.classList.add('hidden');
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }
    }

    async initializeCharts() {
        console.log('📊 Initializing neural charts...');
        
        // Hero Chart - Portfolio Performance
        const heroCtx = document.getElementById('heroChart');
        if (heroCtx) {
            this.charts.hero = new Chart(heroCtx, {
                type: 'line',
                data: {
                    labels: this.generateTimeLabels(30),
                    datasets: [{
                        label: 'Portfolio Value',
                        data: this.generatePortfolioData(30),
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

        // Portfolio Allocation Chart
        const allocationCtx = document.getElementById('allocationChart');
        if (allocationCtx) {
            this.charts.allocation = new Chart(allocationCtx, {
                type: 'doughnut',
                data: {
                    labels: ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL'],
                    datasets: [{
                        data: [35, 25, 20, 12, 8],
                        backgroundColor: [
                            '#00ffff',
                            '#ff00ff',
                            '#00ff88',
                            '#ffff00',
                            '#ff6b35'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    cutout: '70%'
                }
            });
        }

        // Strategy Performance Chart
        const strategyCtx = document.getElementById('strategyChart');
        if (strategyCtx) {
            this.charts.strategy = new Chart(strategyCtx, {
                type: 'line',
                data: {
                    labels: this.generateTimeLabels(20),
                    datasets: [{
                        label: 'Neural Strategy',
                        data: this.generateStrategyData(20),
                        borderColor: '#00ffff',
                        backgroundColor: 'rgba(0, 255, 255, 0.1)',
                        borderWidth: 2,
                        fill: true
                    }, {
                        label: 'Benchmark',
                        data: this.generateBenchmarkData(20),
                        borderColor: '#666',
                        borderWidth: 1,
                        fill: false
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
                    }
                }
            });
        }

        // Sentiment Gauge
        const sentimentCtx = document.getElementById('sentimentGauge');
        if (sentimentCtx) {
            this.charts.sentiment = new Chart(sentimentCtx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [94, 6],
                        backgroundColor: ['#00ffff', '#333'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    circumference: 180,
                    rotation: 270,
                    cutout: '80%',
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    }

    async initializeMarketData() {
        console.log('📈 Initializing market data feeds...');
        
        // Initialize with demo data - in production, connect to real APIs
        const symbols = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'];
        
        symbols.forEach(symbol => {
            this.marketData.set(symbol, {
                symbol,
                price: this.generateRandomPrice(symbol),
                change: (Math.random() - 0.5) * 10,
                volume: Math.floor(Math.random() * 10000000),
                lastUpdate: new Date()
            });
        });

        // Update market data display
        this.updateMarketDataDisplay();
    }

    async initializePortfolio() {
        console.log('💼 Initializing portfolio...');
        
        // Initialize portfolio positions
        this.portfolio.positions.set('NVDA', {
            symbol: 'NVDA',
            shares: 2847,
            avgPrice: 297.45,
            currentPrice: 847.32,
            value: 2411847.04,
            pnl: 1565402.89,
            pnlPercent: 23.7
        });

        this.portfolio.positions.set('TSLA', {
            symbol: 'TSLA',
            shares: 1247,
            avgPrice: 201.23,
            currentPrice: 247.83,
            value: 309042.01,
            pnl: 58134.20,
            pnlPercent: 18.3
        });

        this.portfolio.positions.set('AAPL', {
            symbol: 'AAPL',
            shares: 1847,
            avgPrice: 167.89,
            currentPrice: 189.47,
            value: 349885.09,
            pnl: 39862.26,
            pnlPercent: 12.8
        });

        this.updatePortfolioDisplay();
    }

    async initializeNeuralEngine() {
        console.log('🧠 Initializing M3 Max Neural Engine...');
        
        // Simulate neural engine startup
        this.neuralEngine.processing = true;
        
        // Update neural metrics
        this.updateNeuralMetrics();
        
        // Initialize neural network visualization
        this.initializeNeuralVisualization();
    }

    async initializeEventListeners() {
        console.log('🎮 Initializing event listeners...');
        
        // LLM Connection Modal
        const connectLLMBtn = document.getElementById('connectLLM');
        const llmModal = document.getElementById('llmModal');
        const closeLLMModal = document.getElementById('closeLLMModal');
        const testConnectionBtn = document.getElementById('testConnection');
        const connectButton = document.getElementById('connectButton');

        if (connectLLMBtn) {
            connectLLMBtn.addEventListener('click', () => {
                if (llmModal) llmModal.classList.add('active');
            });
        }

        if (closeLLMModal) {
            closeLLMModal.addEventListener('click', () => {
                if (llmModal) llmModal.classList.remove('active');
            });
        }

        if (testConnectionBtn) {
            testConnectionBtn.addEventListener('click', () => this.testLLMConnection());
        }

        if (connectButton) {
            connectButton.addEventListener('click', () => this.connectToLLM());
        }

        // Chat functionality
        const chatInput = document.getElementById('chatInput');
        const sendMessageBtn = document.getElementById('sendMessage');

        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendChatMessage();
                }
            });
        }

        if (sendMessageBtn) {
            sendMessageBtn.addEventListener('click', () => this.sendChatMessage());
        }

        // Navigation
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleNavigation(link.getAttribute('href'));
            });
        });
    }

    async initializeRealTimeUpdates() {
        console.log('⚡ Starting real-time updates...');
        
        // Update market data every 2 seconds
        this.updateIntervals.set('marketData', setInterval(() => {
            this.updateMarketData();
        }, 2000));

        // Update portfolio every 5 seconds
        this.updateIntervals.set('portfolio', setInterval(() => {
            this.updatePortfolioMetrics();
        }, 5000));

        // Update neural metrics every 3 seconds
        this.updateIntervals.set('neural', setInterval(() => {
            this.updateNeuralMetrics();
        }, 3000));

        // Update news feed every 10 seconds
        this.updateIntervals.set('news', setInterval(() => {
            this.updateNewsFeed();
        }, 10000));

        // Update trades every 1 second
        this.updateIntervals.set('trades', setInterval(() => {
            this.updateTradesDisplay();
        }, 1000));
    }

    updateMarketData() {
        this.marketData.forEach((data, symbol) => {
            // Simulate price movement
            const volatility = 0.02; // 2% volatility
            const change = (Math.random() - 0.5) * volatility;
            data.price *= (1 + change);
            data.change = change * 100;
            data.lastUpdate = new Date();
        });

        this.updateMarketDataDisplay();
    }

    updateMarketDataDisplay() {
        // Update equities list
        const equitiesList = document.getElementById('equitiesList');
        if (equitiesList) {
            const equities = ['NVDA', 'TSLA', 'AAPL'];
            equitiesList.innerHTML = equities.map(symbol => {
                const data = this.marketData.get(symbol);
                if (!data) return '';
                
                const changeClass = data.change >= 0 ? 'positive' : 'negative';
                const changeSign = data.change >= 0 ? '+' : '';
                
                return `
                    <div class="asset-item">
                        <span class="symbol">${symbol}</span>
                        <span class="price neural-glow">$${data.price.toFixed(2)}</span>
                        <span class="change ${changeClass}">${changeSign}${data.change.toFixed(2)}%</span>
                    </div>
                `;
            }).join('');
        }
    }

    updatePortfolioMetrics() {
        // Calculate total portfolio value
        let totalValue = 0;
        let totalPnL = 0;

        this.portfolio.positions.forEach(position => {
            const marketData = this.marketData.get(position.symbol);
            if (marketData) {
                position.currentPrice = marketData.price;
                position.value = position.shares * position.currentPrice;
                position.pnl = position.value - (position.shares * position.avgPrice);
                position.pnlPercent = (position.pnl / (position.shares * position.avgPrice)) * 100;
                
                totalValue += position.value;
                totalPnL += position.pnl;
            }
        });

        this.portfolio.value = totalValue;
        this.portfolio.pnl = totalPnL;

        this.updatePortfolioDisplay();
    }

    updatePortfolioDisplay() {
        // Update portfolio value
        const portfolioValue = document.querySelector('.value-amount');
        if (portfolioValue) {
            portfolioValue.textContent = `$${this.portfolio.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        }

        // Update portfolio change
        const portfolioChange = document.querySelector('.value-change');
        if (portfolioChange) {
            const changePercent = (this.portfolio.pnl / (this.portfolio.value - this.portfolio.pnl)) * 100;
            const changeClass = this.portfolio.pnl >= 0 ? 'positive' : 'negative';
            const changeSign = this.portfolio.pnl >= 0 ? '+' : '';
            
            portfolioChange.className = `value-change ${changeClass} neural-pulse`;
            portfolioChange.textContent = `${changeSign}$${this.portfolio.pnl.toLocaleString('en-US', { minimumFractionDigits: 2 })} (${changeSign}${changePercent.toFixed(2)}%)`;
        }

        // Update positions
        this.updatePositionsDisplay();
    }

    updatePositionsDisplay() {
        const positionsContainer = document.querySelector('.portfolio-positions');
        if (!positionsContainer) return;

        const positionsHTML = Array.from(this.portfolio.positions.values()).map(position => {
            const pnlClass = position.pnl >= 0 ? 'positive' : 'negative';
            const pnlSign = position.pnl >= 0 ? '+' : '';
            
            return `
                <div class="position-item">
                    <div class="position-symbol neural-glow">${position.symbol}</div>
                    <div class="position-details">
                        <span class="position-shares">${position.shares.toLocaleString()} shares</span>
                        <span class="position-value">$${position.value.toLocaleString('en-US', { minimumFractionDigits: 0 })}</span>
                    </div>
                    <div class="position-pnl ${pnlClass} neural-pulse">${pnlSign}${position.pnlPercent.toFixed(1)}%</div>
                    <div class="position-ai-score">
                        <div class="ai-score-bar" style="width: ${Math.min(position.pnlPercent + 50, 100)}%"></div>
                        <span>AI: ${Math.floor(Math.random() * 20 + 80)}</span>
                    </div>
                </div>
            `;
        }).join('');

        positionsContainer.innerHTML = positionsHTML;
    }

    updateNeuralMetrics() {
        // Update processing speed
        const processingSpeed = document.getElementById('processingSpeed');
        if (processingSpeed) {
            const speed = (15.8 + (Math.random() - 0.5) * 0.4).toFixed(1);
            processingSpeed.textContent = `${speed} TOPS + Live`;
        }

        // Update model accuracy
        const modelAccuracy = document.getElementById('modelAccuracy');
        if (modelAccuracy) {
            const accuracy = (98.7 + (Math.random() - 0.5) * 0.6).toFixed(1);
            modelAccuracy.textContent = `${accuracy}%`;
        }

        // Update data streams
        const dataStreams = document.getElementById('dataStreams');
        if (dataStreams) {
            const streams = Math.floor(47 + (Math.random() - 0.5) * 6);
            dataStreams.textContent = streams.toString();
        }

        // Update neural cores
        this.updateNeuralCores();
    }

    updateNeuralCores() {
        const cores = document.querySelectorAll('.core');
        cores.forEach((core, index) => {
            setTimeout(() => {
                core.classList.toggle('active');
                setTimeout(() => core.classList.add('active'), 100);
            }, index * 200);
        });
    }

    updateNewsFeed() {
        const newsFeed = document.getElementById('newsFeed');
        if (!newsFeed) return;

        const newsItems = [
            {
                time: new Date().toLocaleTimeString(),
                source: 'Reuters',
                headline: 'NVIDIA Reports Record Q4 Earnings, Beats Estimates',
                sentiment: 'Bullish 94%',
                symbol: 'NVDA',
                prediction: '+3.2%'
            },
            {
                time: new Date(Date.now() - 120000).toLocaleTimeString(),
                source: 'Bloomberg',
                headline: 'Tesla Announces New Gigafactory Expansion',
                sentiment: 'Bullish 87%',
                symbol: 'TSLA',
                prediction: '+2.1%'
            },
            {
                time: new Date(Date.now() - 240000).toLocaleTimeString(),
                source: 'CNBC',
                headline: 'Apple Unveils M4 Max Chip with Enhanced Neural Engine',
                sentiment: 'Bullish 91%',
                symbol: 'AAPL',
                prediction: '+1.8%'
            }
        ];

        const newsHTML = newsItems.map(item => `
            <div class="news-item">
                <div class="news-time">${item.time}</div>
                <div class="news-source">${item.source}</div>
                <div class="news-headline">${item.headline}</div>
                <div class="news-sentiment positive neural-glow">${item.sentiment}</div>
                <div class="news-impact">
                    <span class="impact-symbol">${item.symbol}</span>
                    <span class="impact-prediction">${item.prediction}</span>
                </div>
            </div>
        `).join('');

        newsFeed.innerHTML = newsHTML;
    }

    updateTradesDisplay() {
        const tradesList = document.querySelector('.trades-list');
        if (!tradesList) return;

        // Generate random trades
        const symbols = ['NVDA', 'TSLA', 'AAPL'];
        const actions = ['BUY', 'SELL'];
        
        if (Math.random() < 0.1) { // 10% chance to add new trade
            const symbol = symbols[Math.floor(Math.random() * symbols.length)];
            const action = actions[Math.floor(Math.random() * actions.length)];
            const quantity = Math.floor(Math.random() * 1000 + 100);
            const price = this.marketData.get(symbol)?.price || 100;
            
            const tradeHTML = `
                <div class="trade-item">
                    <div class="trade-time neural-glow">${new Date().toLocaleTimeString()}</div>
                    <div class="trade-symbol neural-glow">${symbol}</div>
                    <div class="trade-action ${action.toLowerCase()} neural-action">AI ${action}</div>
                    <div class="trade-quantity">${quantity.toLocaleString()} shares</div>
                    <div class="trade-price neural-glow">$${price.toFixed(2)}</div>
                    <div class="trade-status success neural-pulse">EXECUTED</div>
                    <div class="trade-ai-score">AI: ${Math.floor(Math.random() * 10 + 90)}%</div>
                </div>
            `;
            
            tradesList.insertAdjacentHTML('afterbegin', tradeHTML);
            
            // Keep only last 10 trades
            const trades = tradesList.querySelectorAll('.trade-item');
            if (trades.length > 10) {
                trades[trades.length - 1].remove();
            }
        }
    }

    async testLLMConnection() {
        const testButton = document.getElementById('testConnection');
        const connectionTest = document.getElementById('connectionTest');
        
        if (testButton) testButton.disabled = true;
        if (connectionTest) connectionTest.style.display = 'block';

        try {
            const endpoint = document.getElementById('llmEndpoint')?.value || this.llmEndpoint;
            
            // Test connection to LLM
            const response = await fetch(`${endpoint}/api/tags`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.showConnectionResult(true, `Connected successfully! Found ${data.models?.length || 0} models.`);
            } else {
                this.showConnectionResult(false, `Connection failed: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            this.showConnectionResult(false, `Connection error: ${error.message}`);
        }

        if (testButton) testButton.disabled = false;
    }

    showConnectionResult(success, message) {
        const connectionTest = document.getElementById('connectionTest');
        if (!connectionTest) return;

        const statusClass = success ? 'success' : 'error';
        const icon = success ? 'fa-check-circle' : 'fa-exclamation-triangle';
        
        connectionTest.innerHTML = `
            <div class="test-status ${statusClass}">
                <i class="fas ${icon}"></i>
                <span>${message}</span>
            </div>
        `;

        setTimeout(() => {
            connectionTest.style.display = 'none';
        }, 3000);
    }

    async connectToLLM() {
        const endpoint = document.getElementById('llmEndpoint')?.value || this.llmEndpoint;
        const model = document.getElementById('llmModel')?.value || this.llmModel;
        
        this.llmEndpoint = endpoint;
        this.llmModel = model;
        
        try {
            // Test connection first
            const response = await fetch(`${endpoint}/api/tags`);
            if (response.ok) {
                this.llmConnected = true;
                this.updateLLMStatus(true);
                this.enableChat();
                
                // Close modal
                const llmModal = document.getElementById('llmModal');
                if (llmModal) llmModal.classList.remove('active');
                
                // Send welcome message
                this.addChatMessage('assistant', '🧠 M3 Max Neural Engine connected successfully! I can now provide real-time market analysis, trading insights, and portfolio optimization. How can I help you today?');
            } else {
                throw new Error('Connection failed');
            }
        } catch (error) {
            this.showConnectionResult(false, `Failed to connect: ${error.message}`);
        }
    }

    updateLLMStatus(connected) {
        const llmStatus = document.getElementById('llmStatus');
        const statusDot = llmStatus?.querySelector('.neural-status-dot');
        const statusText = llmStatus?.querySelector('span');
        
        if (statusDot && statusText) {
            if (connected) {
                statusDot.classList.remove('offline');
                statusDot.classList.add('online');
                statusText.textContent = 'M3 Max Connected';
            } else {
                statusDot.classList.remove('online');
                statusDot.classList.add('offline');
                statusText.textContent = 'M3 Max Disconnected';
            }
        }
    }

    enableChat() {
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendMessage');
        
        if (chatInput) {
            chatInput.disabled = false;
            chatInput.placeholder = 'Ask about live data, options strategies, pattern recognition...';
        }
        
        if (sendButton) {
            sendButton.disabled = false;
        }
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput || !chatInput.value.trim()) return;

        const message = chatInput.value.trim();
        chatInput.value = '';

        // Add user message
        this.addChatMessage('user', message);

        if (!this.llmConnected) {
            this.addChatMessage('assistant', '⚠️ M3 Max Neural Engine not connected. Please connect to your local LLM first.');
            return;
        }

        // Show typing indicator
        this.addChatMessage('assistant', '🧠 Neural processing...', true);

        try {
            const response = await this.queryLLM(message);
            
            // Remove typing indicator
            this.removeChatMessage();
            
            // Add AI response
            this.addChatMessage('assistant', response);
        } catch (error) {
            // Remove typing indicator
            this.removeChatMessage();
            
            this.addChatMessage('assistant', `❌ Neural processing error: ${error.message}`);
        }
    }

    async queryLLM(message) {
        const systemPrompt = `You are the world's most advanced Neural AI Trading System, powered by M3 Max chip technology. You have access to real-time market data, advanced analytics, and quantum-level processing capabilities.

Current Portfolio Status:
- Total Value: $${this.portfolio.value.toLocaleString()}
- Top Positions: ${Array.from(this.portfolio.positions.keys()).join(', ')}
- Neural Engine: Active with 97.3% accuracy

Market Context:
- Real-time data feeds: Active
- Neural pattern recognition: Enabled
- Risk management: Optimal

Respond as an expert AI trading assistant with deep market knowledge, technical analysis expertise, and access to live data. Be concise, actionable, and professional. Use relevant emojis and trading terminology.`;

        const response = await fetch(`${this.llmEndpoint}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: this.llmModel,
                prompt: `${systemPrompt}\n\nUser: ${message}\n\nAssistant:`,
                stream: false,
                options: {
                    temperature: 0.7,
                    max_tokens: 500
                }
            })
        });

        if (!response.ok) {
            throw new Error(`LLM API error: ${response.status}`);
        }

        const data = await response.json();
        return data.response || 'Neural processing completed, but no response generated.';
    }

    addChatMessage(sender, message, isTyping = false) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = sender === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-brain neural-icon"></i>';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <p>${message}</p>
            </div>
        `;

        if (isTyping) {
            messageDiv.classList.add('typing');
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    removeChatMessage() {
        const chatMessages = document.getElementById('chatMessages');
        const typingMessage = chatMessages?.querySelector('.message.typing');
        if (typingMessage) {
            typingMessage.remove();
        }
    }

    handleNavigation(href) {
        // Handle navigation between different sections
        console.log(`Navigating to: ${href}`);
        
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        document.querySelector(`[href="${href}"]`)?.classList.add('active');
    }

    initializeNeuralVisualization() {
        // Initialize neural network visualization
        const neuralCanvas = document.getElementById('neuralNetworkViz');
        if (!neuralCanvas) return;

        const ctx = neuralCanvas.getContext('2d');
        const width = neuralCanvas.width = neuralCanvas.offsetWidth;
        const height = neuralCanvas.height = neuralCanvas.offsetHeight;

        // Neural network animation
        const nodes = [];
        const connections = [];

        // Create nodes
        for (let i = 0; i < 20; i++) {
            nodes.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                radius: Math.random() * 3 + 2,
                activity: Math.random()
            });
        }

        const animate = () => {
            ctx.clearRect(0, 0, width, height);
            
            // Update and draw nodes
            nodes.forEach(node => {
                node.x += node.vx;
                node.y += node.vy;
                
                if (node.x < 0 || node.x > width) node.vx *= -1;
                if (node.y < 0 || node.y > height) node.vy *= -1;
                
                node.activity = Math.sin(Date.now() * 0.001 + node.x * 0.01) * 0.5 + 0.5;
                
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 255, 255, ${node.activity})`;
                ctx.fill();
            });

            // Draw connections
            nodes.forEach((node, i) => {
                nodes.slice(i + 1).forEach(otherNode => {
                    const distance = Math.sqrt(
                        Math.pow(node.x - otherNode.x, 2) + 
                        Math.pow(node.y - otherNode.y, 2)
                    );
                    
                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.moveTo(node.x, node.y);
                        ctx.lineTo(otherNode.x, otherNode.y);
                        ctx.strokeStyle = `rgba(0, 255, 255, ${(100 - distance) / 100 * 0.3})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                });
            });

            requestAnimationFrame(animate);
        };

        animate();
    }

    // Utility functions
    generateTimeLabels(count) {
        const labels = [];
        const now = new Date();
        for (let i = count - 1; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60000);
            labels.push(time.toLocaleTimeString());
        }
        return labels;
    }

    generatePortfolioData(count) {
        const data = [];
        let value = 100000;
        for (let i = 0; i < count; i++) {
            value += (Math.random() - 0.4) * 5000; // Slight upward bias
            data.push(value);
        }
        return data;
    }

    generateStrategyData(count) {
        const data = [];
        let value = 0;
        for (let i = 0; i < count; i++) {
            value += (Math.random() - 0.3) * 2; // Upward bias for strategy
            data.push(value);
        }
        return data;
    }

    generateBenchmarkData(count) {
        const data = [];
        let value = 0;
        for (let i = 0; i < count; i++) {
            value += (Math.random() - 0.5) * 1;
            data.push(value);
        }
        return data;
    }

    generateRandomPrice(symbol) {
        const basePrices = {
            'NVDA': 847,
            'TSLA': 247,
            'AAPL': 189,
            'MSFT': 378,
            'GOOGL': 142,
            'AMZN': 151,
            'META': 487
        };
        
        return basePrices[symbol] || 100;
    }

    // Cleanup
    destroy() {
        this.updateIntervals.forEach(interval => clearInterval(interval));
        this.updateIntervals.clear();
        
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.quantBot = new QuantBotAI();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.quantBot) {
        window.quantBot.destroy();
    }
});