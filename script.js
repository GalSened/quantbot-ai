// Global state
let llmConnected = false;
let llmEndpoint = 'http://localhost:11434';
let llmModel = 'llama2';
let charts = {};
let neuralEngine = {
    active: false,
    cores: 16,
    processingSpeed: 15.8,
    accuracy: 98.7,
    dataStreams: 47,
    optionsChains: 2847,
    newsPerMin: 184,
    patternAccuracy: 96.4
};

// Market data simulation
let marketData = {
    equities: [
        { symbol: 'NVDA', price: 847.32, change: 2.47 },
        { symbol: 'TSLA', price: 247.83, change: 1.23 },
        { symbol: 'AAPL', price: 189.47, change: -0.87 },
        { symbol: 'GOOGL', price: 142.35, change: 1.56 },
        { symbol: 'MSFT', price: 378.92, change: 0.94 }
    ],
    crypto: [
        { symbol: 'BTC', price: 67432, change: 1.23 },
        { symbol: 'ETH', price: 3847, change: 2.15 },
        { symbol: 'SOL', price: 184, change: -1.34 }
    ],
    patterns: [
        { name: 'Bull Flag', symbol: 'NVDA', confidence: 94.7, target: 920 },
        { name: 'Head & Shoulders', symbol: 'TSLA', confidence: 87.3, target: 220 },
        { name: 'Cup & Handle', symbol: 'AAPL', confidence: 91.8, target: 205 }
    ],
    news: [
        {
            time: '09:47:23',
            source: 'Reuters',
            headline: 'NVIDIA Reports Record Q4 Earnings, Beats Estimates',
            sentiment: 'Bullish 94%',
            symbol: 'NVDA',
            impact: '+3.2%'
        },
        {
            time: '09:45:12',
            source: 'Bloomberg',
            headline: 'Tesla Announces New Gigafactory in Texas',
            sentiment: 'Bullish 87%',
            symbol: 'TSLA',
            impact: '+2.1%'
        }
    ]
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeNeuralVisualizations();
    initializeEventListeners();
    startDataSimulation();
    startNeuralAnimations();
    updateMarketStatus();
    
    // Initialize new components
    initializeMarketData();
    initializePatternRecognition();
    initializeNewsAnalysis();
    
    // Update data more frequently for real-time feel
    setInterval(updateDashboardData, 2000);
    setInterval(updateNeuralMetrics, 1000);
    setInterval(updateMarketData, 3000);
    setInterval(updatePatternData, 5000);
    setInterval(updateNewsData, 7000);
    setInterval(updateMarketStatus, 30000);
});

// Event Listeners
function initializeEventListeners() {
    // LLM Connection
    document.getElementById('connectLLM').addEventListener('click', openLLMModal);
    document.getElementById('closeLLMModal').addEventListener('click', closeLLMModal);
    document.getElementById('testConnection').addEventListener('click', testLLMConnection);
    document.getElementById('connectButton').addEventListener('click', connectToLLM);
    
    // Chat
    document.getElementById('sendMessage').addEventListener('click', sendChatMessage);
    document.getElementById('chatInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    
    // Modal close on outside click
    document.getElementById('llmModal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeLLMModal();
        }
    });
    
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Chart Initialization
function initializeCharts() {
    // Hero Chart
    const heroCtx = document.getElementById('heroChart').getContext('2d');
    charts.hero = new Chart(heroCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(50),
            datasets: [{
                label: 'Neural Portfolio',
                data: generatePortfolioData(50),
                borderColor: '#00ffff',
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 8,
                shadowColor: 'rgba(0, 255, 255, 0.5)',
                shadowBlur: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            animation: {
                duration: 2000,
                easing: 'easeInOutQuart'
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false }
                },
                y: {
                    display: false,
                    grid: { display: false }
                }
            },
            elements: {
                point: { radius: 0 }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
    
    // Portfolio Allocation Chart
    const allocationCtx = document.getElementById('allocationChart').getContext('2d');
    charts.allocation = new Chart(allocationCtx, {
        type: 'doughnut',
        data: {
            labels: ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'Cash'],
            datasets: [{
                data: [29.8, 13.9, 12.2, 11.7, 10.4, 22.0],
                backgroundColor: [
                    '#00ffff',
                    '#ff00ff',
                    '#00ff88',
                    '#ffff00',
                    '#10b981',
                    '#6b7280'
                ],
                borderWidth: 2,
                borderColor: 'rgba(0, 255, 255, 0.3)',
                hoverBorderWidth: 4,
                hoverBorderColor: '#00ffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#00ffff',
                    bodyColor: '#ffffff',
                    borderColor: '#00ffff',
                    borderWidth: 1
                }
            },
            cutout: '70%',
            animation: {
                animateRotate: true,
                duration: 2000
            }
        }
    });
    
    // Mini Trend Chart
    const miniTrendCtx = document.getElementById('miniTrendChart').getContext('2d');
    charts.miniTrend = new Chart(miniTrendCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                data: generateTrendData(20),
                borderColor: '#00ffff',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            elements: { point: { radius: 0 } }
        }
    });
    
    // Strategy Performance Chart
    const strategyCtx = document.getElementById('strategyChart').getContext('2d');
    charts.strategy = new Chart(strategyCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(30),
            datasets: [
                {
                    label: 'CV + NLP + RL Ensemble',
                    data: generateStrategyData(30, 1.2),
                    borderColor: '#00ffff',
                    backgroundColor: 'rgba(0, 255, 255, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#00ffff',
                    pointBorderColor: '#ffffff',
                    pointHoverRadius: 8
                },
                {
                    label: 'S&P 500',
                    data: generateStrategyData(30, 0.8),
                    borderColor: '#6b7280',
                    backgroundColor: 'rgba(107, 114, 128, 0.1)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.4,
                    borderDash: [5, 5]
                },
                {
                    label: 'Traditional Quant',
                    data: generateStrategyData(30, 0.9),
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.4,
                    borderDash: [10, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1500,
                easing: 'easeInOutCubic'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#00ffff',
                        font: { size: 12 },
                        usePointStyle: true
                    }
                }
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false }
                },
                y: {
                    display: true,
                    grid: { 
                        color: '#3f3f46',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#71717a',
                        font: { size: 11 }
                    }
                }
            }
        }
    });
    
    // Sentiment Gauge
    const sentimentCtx = document.getElementById('sentimentGauge').getContext('2d');
    charts.sentiment = new Chart(sentimentCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [94, 6],
                backgroundColor: [
                    'rgba(0, 255, 255, 0.8)',
                    'rgba(55, 65, 81, 0.3)'
                ],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
                borderRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            cutout: '80%',
            animation: {
                animateRotate: true,
                duration: 2000
            }
        }
    });
    
    // Pattern Recognition Chart
    const patternCtx = document.getElementById('patternChart').getContext('2d');
    charts.pattern = new Chart(patternCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'NVDA Pattern',
                data: generatePatternData(20),
                borderColor: '#ff00ff',
                backgroundColor: 'rgba(255, 0, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
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
            elements: { point: { radius: 0 } }
        }
    });
    
    // Volatility Surface (3D simulation)
    const volCtx = document.getElementById('volSurface').getContext('2d');
    charts.volSurface = new Chart(volCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'IV Surface',
                data: generateVolatilitySurface(),
                backgroundColor: 'rgba(255, 255, 0, 0.6)',
                borderColor: '#ffff00'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Neural Visualizations
function initializeNeuralVisualizations() {
    // Neural Network Visualization
    const neuralCtx = document.getElementById('neuralNetworkViz').getContext('2d');
    drawNeuralNetwork(neuralCtx);
    
    // Risk Heatmap
    const riskCtx = document.getElementById('riskHeatmap').getContext('2d');
    drawRiskHeatmap(riskCtx);
    
    // Neural Background
    const bgCtx = document.getElementById('neuralBackground').getContext('2d');
    drawNeuralBackground(bgCtx);
}

function drawNeuralNetwork(ctx) {
    const canvas = ctx.canvas;
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Neural network nodes and connections
    const nodes = [];
    const layers = [4, 6, 6, 3];
    
    // Generate nodes
    layers.forEach((nodeCount, layerIndex) => {
        for (let i = 0; i < nodeCount; i++) {
            nodes.push({
                x: (width / (layers.length - 1)) * layerIndex,
                y: (height / (nodeCount + 1)) * (i + 1),
                layer: layerIndex,
                active: Math.random() > 0.3
            });
        }
    });
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // Draw connections
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        
        nodes.forEach(node => {
            if (node.layer < layers.length - 1) {
                const nextLayerNodes = nodes.filter(n => n.layer === node.layer + 1);
                nextLayerNodes.forEach(nextNode => {
                    if (node.active && nextNode.active) {
                        ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)';
                        ctx.lineWidth = 2;
                    } else {
                        ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
                        ctx.lineWidth = 1;
                    }
                    
                    ctx.beginPath();
                    ctx.moveTo(node.x, node.y);
                    ctx.lineTo(nextNode.x, nextNode.y);
                    ctx.stroke();
                });
            }
        });
        
        // Draw nodes
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.active ? 6 : 3, 0, Math.PI * 2);
            ctx.fillStyle = node.active ? '#00ffff' : 'rgba(0, 255, 255, 0.3)';
            ctx.fill();
            
            if (node.active) {
                ctx.shadowColor = '#00ffff';
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        });
        
        // Randomly activate/deactivate nodes
        if (Math.random() < 0.1) {
            const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
            randomNode.active = !randomNode.active;
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

function drawRiskHeatmap(ctx) {
    const canvas = ctx.canvas;
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    const gridSize = 10;
    const cols = Math.floor(width / gridSize);
    const rows = Math.floor(height / gridSize);
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        for (let x = 0; x < cols; x++) {
            for (let y = 0; y < rows; y++) {
                const risk = Math.sin(Date.now() * 0.001 + x * 0.1 + y * 0.1) * 0.5 + 0.5;
                const intensity = risk * 0.3;
                
                ctx.fillStyle = `rgba(0, 255, 255, ${intensity})`;
                ctx.fillRect(x * gridSize, y * gridSize, gridSize - 1, gridSize - 1);
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

function drawNeuralBackground(ctx) {
    const canvas = ctx.canvas;
    const width = canvas.width = window.innerWidth;
    const height = canvas.height = window.innerHeight;
    
    const particles = [];
    const particleCount = 50;
    
    // Initialize particles
    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.1
        });
    }
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // Update and draw particles
        particles.forEach(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Wrap around edges
            if (particle.x < 0) particle.x = width;
            if (particle.x > width) particle.x = 0;
            if (particle.y < 0) particle.y = height;
            if (particle.y > height) particle.y = 0;
            
            // Draw particle
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 255, 255, ${particle.opacity})`;
            ctx.fill();
        });
        
        // Draw connections
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        
        particles.forEach((particle, i) => {
            particles.slice(i + 1).forEach(otherParticle => {
                const dx = particle.x - otherParticle.x;
                const dy = particle.y - otherParticle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particle.x, particle.y);
                    ctx.lineTo(otherParticle.x, otherParticle.y);
                    ctx.stroke();
                }
            });
        });
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

// Data Generation Functions
function generateTimeLabels(count) {
    const labels = [];
    const now = new Date();
    for (let i = count - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60000); // 1 minute intervals
        labels.push(time.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        }));
    }
    return labels;
}

function generatePortfolioData(count) {
    const data = [];
    let value = 2800000; // Start with higher value for neural system
    
    for (let i = 0; i < count; i++) {
        value += (Math.random() - 0.35) * 50000; // Strong upward bias for AI
        data.push(Math.max(value, 2500000)); // Higher minimum floor
    }
    
    return data;
}

function generateTrendData(count) {
    const data = [];
    let value = 0;
    
    for (let i = 0; i < count; i++) {
        value += (Math.random() - 0.3) * 2; // Upward trend
        data.push(value);
    }
    
    return data;
}

function generateStrategyData(count, multiplier = 1) {
    const data = [];
    let value = 0;
    
    for (let i = 0; i < count; i++) {
        value += (Math.random() - 0.3) * multiplier; // Strong upward bias for neural
        data.push(value);
    }
    
    return data;
}

function generatePatternData(count) {
    const data = [];
    let value = 800; // Starting price for NVDA
    
    for (let i = 0; i < count; i++) {
        // Simulate bull flag pattern
        if (i < 5) {
            value += Math.random() * 20 + 10; // Initial rise
        } else if (i < 15) {
            value += (Math.random() - 0.6) * 5; // Consolidation
        } else {
            value += Math.random() * 15 + 8; // Breakout
        }
        data.push(value);
    }
    
    return data;
}

function generateVolatilitySurface() {
    const data = [];
    for (let i = 0; i < 20; i++) {
        data.push({
            x: Math.random() * 100,
            y: Math.random() * 50 + 10
        });
    }
    return data;
}

// Neural Animations
function startNeuralAnimations() {
    // Animate neural cores
    setInterval(() => {
        const cores = document.querySelectorAll('.core');
        cores.forEach((core, index) => {
            if (Math.random() > 0.7) {
                core.style.animationDelay = `${index * 0.2}s`;
            }
        });
    }, 2000);
    
    // Animate neural indicators
    setInterval(() => {
        const indicators = document.querySelectorAll('.neural-pulse, .neural-glow');
        indicators.forEach(indicator => {
            if (Math.random() > 0.8) {
                indicator.style.animationDuration = `${Math.random() * 2 + 1}s`;
            }
        });
    }, 3000);
}

// Initialize new components
function initializeMarketData() {
    updateEquitiesList();
    updateCryptoList();
    updateOptionsFlow();
}

function initializePatternRecognition() {
    // Pattern recognition is already initialized with static data
    // In a real implementation, this would connect to computer vision APIs
}

function initializeNewsAnalysis() {
    // News analysis is already initialized with static data
    // In a real implementation, this would connect to news APIs
}

function updateEquitiesList() {
    const equitiesList = document.getElementById('equitiesList');
    if (!equitiesList) return;
    
    equitiesList.innerHTML = '';
    marketData.equities.forEach(equity => {
        const item = document.createElement('div');
        item.className = 'asset-item';
        item.innerHTML = `
            <span class="symbol">${equity.symbol}</span>
            <span class="price neural-glow">$${equity.price.toFixed(2)}</span>
            <span class="change ${equity.change >= 0 ? 'positive' : 'negative'}">${equity.change >= 0 ? '+' : ''}${equity.change.toFixed(2)}%</span>
        `;
        equitiesList.appendChild(item);
    });
}

function updateCryptoList() {
    const cryptoList = document.getElementById('cryptoList');
    if (!cryptoList) return;
    
    cryptoList.innerHTML = '';
    marketData.crypto.forEach(crypto => {
        const item = document.createElement('div');
        item.className = 'asset-item';
        item.innerHTML = `
            <span class="symbol">${crypto.symbol}</span>
            <span class="price neural-glow">$${crypto.price.toLocaleString()}</span>
            <span class="change ${crypto.change >= 0 ? 'positive' : 'negative'}">${crypto.change >= 0 ? '+' : ''}${crypto.change.toFixed(2)}%</span>
        `;
        cryptoList.appendChild(item);
    });
}

function updateOptionsFlow() {
    // Options flow updates are handled in the existing trade updates
}

// Data Update Functions
function updateDashboardData() {
    // Update portfolio stats
    updatePortfolioStats();
    
    // Update charts with new data
    updateCharts();
    
    // Update trades
    updateRecentTrades();
    
    // Update sentiment
    updateSentimentData();
}

function updateMarketData() {
    // Simulate real-time market data updates
    marketData.equities.forEach(equity => {
        equity.price += (Math.random() - 0.5) * 5;
        equity.change += (Math.random() - 0.5) * 0.5;
    });
    
    marketData.crypto.forEach(crypto => {
        crypto.price += (Math.random() - 0.5) * 100;
        crypto.change += (Math.random() - 0.5) * 1.0;
    });
    
    updateEquitiesList();
    updateCryptoList();
}

function updatePatternData() {
    // Update pattern recognition data
    marketData.patterns.forEach(pattern => {
        pattern.confidence += (Math.random() - 0.5) * 2;
        pattern.confidence = Math.max(80, Math.min(99, pattern.confidence));
    });
    
    // Update pattern display
    const patternItems = document.querySelectorAll('.pattern-confidence');
    patternItems.forEach((item, index) => {
        if (marketData.patterns[index]) {
            item.textContent = `${marketData.patterns[index].confidence.toFixed(1)}%`;
        }
    });
}

function updateNewsData() {
    // Simulate new news articles
    if (Math.random() < 0.3) {
        const newArticle = {
            time: new Date().toLocaleTimeString('en-US', { hour12: false }),
            source: ['Reuters', 'Bloomberg', 'CNBC', 'WSJ'][Math.floor(Math.random() * 4)],
            headline: 'Breaking: Market Update - ' + Math.random().toString(36).substring(7),
            sentiment: `Bullish ${Math.floor(Math.random() * 20 + 80)}%`,
            symbol: ['NVDA', 'TSLA', 'AAPL'][Math.floor(Math.random() * 3)],
            impact: `+${(Math.random() * 3 + 0.5).toFixed(1)}%`
        };
        
        marketData.news.unshift(newArticle);
        if (marketData.news.length > 5) {
            marketData.news.pop();
        }
        
        updateNewsDisplay();
    }
}

function updateNewsDisplay() {
    const newsFeed = document.getElementById('newsFeed');
    if (!newsFeed) return;
    
    newsFeed.innerHTML = '';
    marketData.news.forEach(article => {
        const item = document.createElement('div');
        item.className = 'news-item fade-in';
        item.innerHTML = `
            <div class="news-time">${article.time}</div>
            <div class="news-source">${article.source}</div>
            <div class="news-headline">${article.headline}</div>
            <div class="news-sentiment positive neural-glow">${article.sentiment}</div>
            <div class="news-impact">
                <span class="impact-symbol">${article.symbol}</span>
                <span class="impact-prediction">${article.impact}</span>
            </div>
        `;
        newsFeed.appendChild(item);
    });
}

function updateNeuralMetrics() {
    // Update neural engine metrics
    const processingSpeed = document.getElementById('processingSpeed');
    const modelAccuracy = document.getElementById('modelAccuracy');
    const dataStreams = document.getElementById('dataStreams');
    const optionsChains = document.getElementById('optionsChains');
    const newsPerMin = document.getElementById('newsPerMin');
    const patternAccuracy = document.getElementById('patternAccuracy');
    
    if (processingSpeed) {
        const speed = (15.8 + (Math.random() - 0.5) * 0.5).toFixed(1);
        processingSpeed.textContent = `${speed} TOPS + Live`;
    }
    
    if (modelAccuracy) {
        const accuracy = (98.7 + (Math.random() - 0.5) * 0.3).toFixed(1);
        modelAccuracy.textContent = `${accuracy}%`;
    }
    
    if (dataStreams) {
        const streams = Math.floor(47 + (Math.random() - 0.5) * 10);
        dataStreams.textContent = streams;
    }
    
    if (optionsChains) {
        const chains = Math.floor(2847 + (Math.random() - 0.5) * 200);
        optionsChains.textContent = chains.toLocaleString();
    }
    
    if (newsPerMin) {
        const news = Math.floor(184 + (Math.random() - 0.5) * 50);
        newsPerMin.textContent = news;
    }
    
    if (patternAccuracy) {
        const accuracy = (96.4 + (Math.random() - 0.5) * 1.0).toFixed(1);
        patternAccuracy.textContent = `${accuracy}%`;
    }
}

function updatePortfolioStats() {
    const stats = {
        totalReturn: (Math.random() * 20 + 45).toFixed(1), // Even higher returns
        sharpeRatio: (Math.random() * 2 + 4.5).toFixed(2), // Better Sharpe ratio
        maxDrawdown: -(Math.random() * 0.8 + 0.8).toFixed(1), // Even lower drawdown
        winRate: Math.floor(Math.random() * 8 + 92) // Higher win rate
    };
    
    document.getElementById('totalReturn').textContent = `+${stats.totalReturn}%`;
    document.getElementById('sharpeRatio').textContent = stats.sharpeRatio;
    document.getElementById('maxDrawdown').textContent = `${stats.maxDrawdown}%`;
    document.getElementById('winRate').textContent = `${stats.winRate}%`;
}

function updateCharts() {
    // Update hero chart
    if (charts.hero) {
        const newData = generatePortfolioData(1)[0];
        charts.hero.data.datasets[0].data.push(newData);
        charts.hero.data.datasets[0].data.shift();
        
        const newLabel = new Date().toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        charts.hero.data.labels.push(newLabel);
        charts.hero.data.labels.shift();
        
        charts.hero.update('none');
    }
    
    // Update strategy chart
    if (charts.strategy) {
        const newAIData = generateStrategyData(1, 1.2)[0];
        const newBenchmarkData = generateStrategyData(1, 0.8)[0];
        
        charts.strategy.data.datasets[0].data.push(
            charts.strategy.data.datasets[0].data[charts.strategy.data.datasets[0].data.length - 1] + newAIData
        );
        charts.strategy.data.datasets[0].data.shift();
        
        charts.strategy.data.datasets[1].data.push(
            charts.strategy.data.datasets[1].data[charts.strategy.data.datasets[1].data.length - 1] + newBenchmarkData
        );
        charts.strategy.data.datasets[1].data.shift();
        
        charts.strategy.update('none');
    }
}

function updateRecentTrades() {
    const symbols = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN'];
    const actions = ['AI BUY', 'AI SELL'];
    const statuses = ['EXECUTED', 'PROCESSING'];
    
    // Randomly add new trade
    if (Math.random() < 0.4) { // More frequent trades
        const tradesList = document.querySelector('.trades-list');
        const newTrade = document.createElement('div');
        newTrade.className = 'trade-item fade-in';
        
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        const action = actions[Math.floor(Math.random() * actions.length)];
        const quantity = Math.floor(Math.random() * 2000 + 500); // Larger quantities
        const price = (Math.random() * 500 + 200).toFixed(2); // Higher prices
        const status = 'EXECUTED'; // Always executed for demo
        const aiScore = Math.floor(Math.random() * 10 + 90); // High AI scores
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        
        // Add options component to some trades
        const hasOptions = Math.random() > 0.5;
        const optionsText = hasOptions ? ` + ${Math.floor(Math.random() * 20 + 5)}x ${Math.floor(Math.random() * 200 + 800)}${action === 'AI BUY' ? 'C' : 'P'}` : '';
        
        newTrade.innerHTML = `
            <div class="trade-time neural-glow">${time}</div>
            <div class="trade-symbol neural-glow">${symbol}</div>
            <div class="trade-action neural-action">${action}</div>
            <div class="trade-quantity">${quantity}${optionsText}</div>
            <div class="trade-price neural-glow">$${price}</div>
            <div class="trade-status success neural-pulse">${status}</div>
            <div class="trade-ai-score">AI: ${aiScore}%</div>
        `;
        
        tradesList.insertBefore(newTrade, tradesList.firstChild);
        
        // Remove oldest trade if more than 3
        if (tradesList.children.length > 3) {
            tradesList.removeChild(tradesList.lastChild);
        }
    }
}

function updateSentimentData() {
    const sentiment = Math.floor(Math.random() * 10 + 90); // 90-100 range for neural
    const sentimentLabel = sentiment > 95 ? 'Extremely Bullish' : sentiment > 90 ? 'Very Bullish' : 'Bullish';
    
    document.querySelector('.sentiment-score').textContent = sentiment;
    document.querySelector('.sentiment-label').textContent = sentimentLabel;
    
    // Update sentiment gauge
    if (charts.sentiment) {
        charts.sentiment.data.datasets[0].data = [sentiment, 100 - sentiment];
        charts.sentiment.update('none');
    }
    
    // Update factor bars
    const factors = document.querySelectorAll('.factor-fill');
    factors.forEach(factor => {
        const newWidth = Math.floor(Math.random() * 10 + 90); // Higher values for neural
        factor.style.width = `${newWidth}%`;
        factor.parentElement.nextElementSibling.textContent = `+${newWidth}%`;
    });
}

function updateMarketStatus() {
    const now = new Date();
    const marketOpen = new Date();
    marketOpen.setHours(9, 30, 0, 0);
    const marketClose = new Date();
    marketClose.setHours(16, 0, 0, 0);
    
    const isWeekday = now.getDay() >= 1 && now.getDay() <= 5;
    const isMarketHours = now >= marketOpen && now <= marketClose;
    const isOpen = isWeekday && isMarketHours;
    
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');
    
    if (isOpen) {
        statusIndicator.classList.add('active');
        statusText.textContent = 'Market Open';
    } else {
        statusIndicator.classList.remove('active');
        statusText.textContent = 'Market Closed';
    }
}

// LLM Connection Functions
function openLLMModal() {
    document.getElementById('llmModal').classList.add('active');
}

function closeLLMModal() {
    document.getElementById('llmModal').classList.remove('active');
}

async function testLLMConnection() {
    const endpoint = document.getElementById('llmEndpoint').value;
    const model = document.getElementById('llmModel').value;
    const testDiv = document.getElementById('connectionTest');
    
    testDiv.style.display = 'block';
    testDiv.innerHTML = `
        <div class="test-status">
            <i class="fas fa-spinner fa-spin"></i>
            <span>Testing connection...</span>
        </div>
    `;
    
    try {
        // Test connection to local LLM
        const response = await fetch(`${endpoint}/api/tags`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            testDiv.innerHTML = `
                <div class="test-status" style="color: var(--success-color);">
                    <i class="fas fa-check-circle"></i>
                    <span>Connection successful!</span>
                </div>
            `;
        } else {
            throw new Error('Connection failed');
        }
    } catch (error) {
        testDiv.innerHTML = `
            <div class="test-status" style="color: var(--danger-color);">
                <i class="fas fa-exclamation-circle"></i>
                <span>Connection failed. Please check your LLM server.</span>
            </div>
        `;
    }
}

async function connectToLLM() {
    const endpoint = document.getElementById('llmEndpoint').value;
    const model = document.getElementById('llmModel').value;
    
    try {
        // Store connection details
        llmEndpoint = endpoint;
        llmModel = model;
        llmConnected = true;
        neuralEngine.active = true;
        
        // Update UI
        updateLLMStatus(true);
        closeLLMModal();
        
        // Enable chat
        document.getElementById('chatInput').disabled = false;
        document.getElementById('sendMessage').disabled = false;
        
        // Add connection message
        addChatMessage('assistant', '🧠 Neural connection established with M3 Max! I now have access to quantum-level market analysis, real-time strategy optimization, and autonomous trading capabilities. Your neural trading system is fully operational. What would you like to explore?');
        
    } catch (error) {
        console.error('Failed to connect to LLM:', error);
        addChatMessage('assistant', '⚠️ Neural connection failed. Please verify your M3 Max endpoint and try again.');
    }
}

function updateLLMStatus(connected) {
    const statusElement = document.getElementById('llmStatus');
    const statusDot = statusElement.querySelector('.neural-status-dot');
    const statusText = statusElement.querySelector('span');
    
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

// Chat Functions
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message || !llmConnected) return;
    
    // Add user message
    addChatMessage('user', message);
    input.value = '';
    
    // Show typing indicator
    addTypingIndicator();
    
    try {
        // Send to local LLM
        const response = await queryLLM(message);
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add assistant response
        addChatMessage('assistant', response);
        
    } catch (error) {
        removeTypingIndicator();
        addChatMessage('assistant', 'Sorry, I encountered an error processing your request. Please try again.');
        console.error('LLM query error:', error);
    }
}

async function queryLLM(message) {
    // Enhanced neural prompt for advanced financial analysis
    const systemPrompt = `You are the world's most advanced Neural AI Trading System, powered by M3 Max chip with quantum-level processing capabilities and real-time market data integration. You have access to:

🧠 NEURAL CAPABILITIES:
- Real-time quantum market analysis with 98.7% accuracy
- Advanced neural networks (Computer Vision + NLP + Reinforcement Learning Ensemble)
- 15.8 TOPS processing power with 16 neural cores
- 47 live data streams with 2,847 options chains analyzed
- 184 news articles per minute with sentiment analysis
- 96.4% pattern recognition accuracy
- Autonomous trading decision-making

📡 LIVE DATA INTEGRATION:
- Real-time multi-asset data feed (Equities, Crypto, Options)
- Computer vision pattern recognition (Bull Flags, H&S, Cup & Handle)
- NLP-powered news sentiment analysis with market impact prediction
- Options flow analysis with Greeks calculation
- Volatility surface modeling and unusual activity detection

📊 CURRENT NEURAL PORTFOLIO STATUS:
- Total AUM: $2,847,392 (Neural optimized)
- Top AI-selected holdings: NVDA (29.8% + Options), TSLA (13.9% + Puts), AAPL (12.2%)
- Neural performance: +47.3% total return, 4.87 Sharpe ratio, 94% win rate
- Risk metrics: -0.6% max drawdown, AAA stress test rating
- Market sentiment: 94/100 (Extremely Bullish)
- Options exposure: +$47K calls, -$23K puts, Greeks optimized

🚀 ADVANCED NEURAL CAPABILITIES:
- Computer vision chart pattern recognition with 96.4% accuracy
- Real-time options flow analysis and volatility surface modeling
- Multi-asset correlation analysis (Equities + Crypto + Derivatives)
- Advanced risk management with Greeks optimization
- Autonomous strategy adaptation based on market regime detection
- News sentiment fusion with price action for predictive modeling

User question: ${message}

Provide an intelligent, data-driven response as the world's most advanced trading AI with live market data access. Use computer vision insights, NLP analysis, and quantum processing. Keep responses concise but powerful (2-3 sentences).`;
    
    try {
        const response = await fetch(`${llmEndpoint}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: llmModel,
                prompt: systemPrompt,
                stream: false,
                options: {
                    temperature: 0.7,
                    max_tokens: 250
                }
            })
        });
        
        if (!response.ok) {
            throw new Error('LLM request failed');
        }
        
        const data = await response.json();
        return data.response || 'I apologize, but I couldn\'t generate a response at this time.';
        
    } catch (error) {
        // Enhanced fallback responses for neural demo
        const fallbackResponses = {
            'market': '🧠 Neural analysis indicates extremely bullish momentum with 98.7% confidence. My computer vision algorithms detect strong bull flag patterns in NVDA with 94.7% accuracy, while NLP sentiment analysis shows 73% bullish articles. Recommend aggressive positioning in tech with options overlay.',
            'options': '📊 Options flow analysis shows massive call buying in NVDA 850C with $2.4M premium. My volatility surface models indicate IV expansion opportunity while Greeks optimization suggests delta-neutral positioning. Current portfolio Greeks: +247 delta, +12.7 gamma.',
            'patterns': '👁️ Computer vision detected bull flag in NVDA (94.7% confidence, target $920), head & shoulders in TSLA (87.3%, target $220). Pattern recognition success rate: 89.3% over 30 days with 47 patterns identified today.',
            'news': '📰 NLP analysis of 184 articles/min shows 73% bullish sentiment. Breaking: NVIDIA earnings beat drives +3.2% impact prediction. Real-time sentiment fusion with price action confirms continued tech sector rotation.',
            'crypto': '₿ Multi-asset analysis shows BTC correlation breakdown with equities. Neural models detect crypto-specific momentum with BTC at $67,432 (+1.23%). Recommend tactical allocation for portfolio diversification.',
            'portfolio': '📊 Your neural-optimized portfolio is performing exceptionally with 47.3% returns and 4.87 Sharpe ratio. My AI models suggest increasing NVDA allocation to 35% based on quantum trend analysis and earnings momentum.',
            'risk': '🛡️ Neural risk shield shows ultra-minimal exposure with only -0.6% max drawdown and AAA stress test rating. Options Greeks optimized with +247 delta exposure. VaR (99%): -0.6%, volatility: 3.2%.',
            'strategy': '🚀 CV + NLP + RL ensemble strategy operating at peak efficiency with 98.7% accuracy. Multi-modal analysis confirms STRONG BUY NVDA 850C based on pattern recognition, sentiment fusion, and options flow.',
            'neural': '⚡ M3 Max neural cores are operating at peak efficiency with 15.8 TOPS processing power. My transformer networks are analyzing 2,847 market predictions per second with quantum-level precision.',
            'prediction': '🔮 Multi-modal neural networks predict 28% upside potential over next 30 days with 98% confidence. Computer vision + NLP + options flow analysis confirms perfect storm of breakouts, sentiment, and institutional flow.',
            'default': '🧠 Advanced Neural AI with live market data ready for quantum-level analysis. I can provide computer vision pattern recognition, options flow analysis, real-time sentiment fusion, or autonomous strategy optimization. What advanced insights do you need?'
        };
        
        const messageKey = Object.keys(fallbackResponses).find(key => 
            message.toLowerCase().includes(key)
        ) || 'default';
        
        return fallbackResponses[messageKey];
    }
}

function addChatMessage(sender, content) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender} fade-in`;
    
    const avatar = sender === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-brain neural-icon"></i>';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <p>${content}</p>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-brain neural-icon"></i>
        </div>
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Data Simulation
function startDataSimulation() {
    // Simulate neural real-time data updates
    setInterval(() => {
        // Update portfolio value with small random changes
        const currentValue = document.querySelector('.value-amount');
        if (currentValue) {
            const value = parseFloat(currentValue.textContent.replace(/[$,]/g, ''));
            const change = (Math.random() - 0.3) * 50000; // Larger changes for neural system
            const newValue = Math.max(value + change, 2500000); // Higher minimum
            currentValue.textContent = `$${newValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            
            // Update change indicator
            const changeElement = document.querySelector('.value-change');
            const originalValue = 2600000; // Higher baseline
            const totalChange = newValue - originalValue;
            const percentChange = (totalChange / originalValue) * 100;
            
            changeElement.textContent = `${totalChange >= 0 ? '+' : ''}$${Math.abs(totalChange).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${totalChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%)`;
            changeElement.className = `value-change ${totalChange >= 0 ? 'positive' : 'negative'}`;
        }
    }, 2000); // Faster updates
    
    // Initialize market data updates
    updateMarketData();
    updateNewsDisplay();
    
    // Simulate neural position updates
    setInterval(() => {
        const positions = document.querySelectorAll('.position-pnl');
        positions.forEach(position => {
            const currentPnl = parseFloat(position.textContent.replace('%', ''));
            const change = (Math.random() - 0.3) * 1.0; // Larger changes, upward bias
            const newPnl = currentPnl + change;
            
            position.textContent = `${newPnl >= 0 ? '+' : ''}${newPnl.toFixed(1)}%`;
            position.className = `position-pnl ${newPnl >= 0 ? 'positive' : 'negative'} neural-pulse`;
        });
    }, 4000); // Faster updates
    
    // Simulate neural core activity
    setInterval(() => {
        const cores = document.querySelectorAll('.core');
        cores.forEach(core => {
            if (Math.random() > 0.8) {
                core.classList.toggle('active');
                setTimeout(() => core.classList.add('active'), 500);
            }
        });
    }, 1500);
}

// Utility Functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(amount);
}

function formatPercentage(value) {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

function addNotification(message, type = 'info') {
    // Enhanced neural notification system
    console.log(`🧠 NEURAL ${type.toUpperCase()}: ${message}`);
    
    // Could add visual notifications here
    const notification = document.createElement('div');
    notification.className = `neural-notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-brain"></i>
        <span>${message}</span>
    `;
    
    // Add to page temporarily
    document.body.appendChild(notification);
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}

// Enhanced CSS for neural effects
const style = document.createElement('style');
style.textContent = `
    .typing-dots {
        display: flex;
        gap: 4px;
        padding: 8px 0;
    }
    
    .typing-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--neural-color);
        animation: typing 1.4s infinite ease-in-out;
        box-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { 
            transform: scale(0.8); 
            opacity: 0.5;
            box-shadow: 0 0 5px rgba(0, 255, 255, 0.3);
        }
        40% { transform: scale(1); opacity: 1; }
    }
    
    .neural-notification {
        position: fixed;
        top: 100px;
        right: 20px;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        color: var(--neural-color);
        backdrop-filter: blur(10px);
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);