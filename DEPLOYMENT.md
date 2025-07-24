```
# 🚀 Deployment Guide - QuantBot AI Neural Trading System

## GitHub Pages Deployment

### Quick Setup (5 minutes)

1. **Create GitHub Repository**
   ```bash
   # Create new repository on GitHub
   # Clone this project or fork it
   git clone https://github.com/yourusername/quantbot-ai.git
   cd quantbot-ai
```

2. **Enable GitHub Pages**

   - Go to repository **Settings**
   - Navigate to **Pages** section
   - Select **Deploy from a branch**
   - Choose **main** branch and **/ (root)** folder
   - Click **Save**
3. **Access Your Site**

   - Your site will be live at: `https://yourusername.github.io/quantbot-ai`
   - Usually takes 2-5 minutes to deploy

### Advanced Deployment Options

#### Option 1: Direct GitHub Pages

```bash
# Push to main branch
git add .
git commit -m "Deploy QuantBot AI Neural Trading System"
git push origin main
```

#### Option 2: GitHub Actions (Automated)

The included `.github/workflows/deploy.yml` will:

- Automatically minify HTML, CSS, and JavaScript
- Optimize assets for production
- Deploy to GitHub Pages on every push to main

#### Option 3: Custom Domain

1. Add `CNAME` file with your domain:
   ```
   yourdomain.com
   ```
2. Configure DNS settings with your domain provider
3. Enable HTTPS in GitHub Pages settings

## Local Development

### Prerequisites

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local LLM server (Ollama recommended)
- Python 3.7+ or Node.js (for local server)

### Setup Local LLM (M3 Max Optimized)

#### Ollama Setup (Recommended)

```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Start server
ollama serve

# Pull neural models
ollama pull llama2      # M3 Max optimized
ollama pull codellama   # For code analysis
ollama pull mistral     # Advanced reasoning
```

#### Alternative LLM Servers

- **LM Studio**: Download from lmstudio.ai
- **Text Generation WebUI**: GitHub.com/oobabooga/text-generation-webui
- **Custom APIs**: Configure endpoint in the app

### Local Development Server

#### Python Server

```bash
python -m http.server 8000
# Open http://localhost:8000
```

#### Node.js Server

```bash
npx serve .
# Open http://localhost:3000
```

#### PHP Server

```bash
php -S localhost:8000
# Open http://localhost:8000
```

## Configuration

### M3 Max Neural Connection

1. Start your local LLM server
2. Click **"Connect M3 Max"** button in the app
3. Configure endpoint (default: `http://localhost:11434`)
4. Select neural model
5. Test connection and connect

### Environment Variables (Optional)

Create `.env` file for advanced configuration:

```env
NEURAL_ENDPOINT=http://localhost:11434
DEFAULT_MODEL=llama2
NEURAL_KEYS=your_api_keys_here
```

## Performance Optimization

### For M3 Max Users

- Use Ollama with Metal acceleration
- Enable GPU acceleration in browser
- Use Safari for best M3 Max integration
- Close unnecessary browser tabs

### General Optimization

- Use modern browser with hardware acceleration
- Ensure stable internet connection for real-time updates
- Use SSD storage for better performance
- 8GB+ RAM recommended for smooth operation

## Troubleshooting

### Common Issues

#### LLM Connection Failed

- Verify LLM server is running
- Check firewall settings
- Try different endpoint URL
- Restart LLM server

#### GitHub Pages Not Updating

- Check Actions tab for deployment status
- Clear browser cache
- Wait 5-10 minutes for propagation
- Verify main branch has latest changes

#### Performance Issues

- Disable browser extensions
- Clear browser cache and cookies
- Check system resources (CPU, RAM)
- Try different browser

### Support

- Check browser console for errors
- Review network tab for failed requests
- Ensure all dependencies are loaded
- Test with different LLM models

## Security Considerations

- All processing happens locally (no data sent to external servers)
- M3 Max neural processing is completely private
- No backend dependencies or external APIs required
- CORS headers configured for local LLM access

## Advanced Features

### Custom Styling

Modify CSS variables in `styles.css`:

```css
:root {
    --neural-color: #00ffff;
    --quantum-color: #ff00ff;
    --accent-color: #00ff88;
}
```

### Data Sources

Connect real market data APIs by modifying `script.js`:

```javascript
// Add your API keys and endpoints
const ALPHA_VANTAGE_KEY = 'your_key';
const POLYGON_API_KEY = 'your_key';
```

### Neural Prompts

Customize AI behavior in the `queryLLM` function:

```javascript
const systemPrompt = `You are the world's most advanced...`;
```

---

**🧠 Neural Trading System Ready for Deployment!**

*Powered by M3 Max • Real-time Analysis • Advanced AI • Professional Grade*

```

```
