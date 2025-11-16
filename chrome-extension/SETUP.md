# Quick Setup Guide

## ✅ Fixed Issues

1. **Icons are now optional** - The extension will load without icon files
2. **API endpoint added** - The extension now uses `/api/check` JSON endpoint
3. **CORS enabled** - Backend now supports cross-origin requests

## Installation Steps

### 1. Install Flask-CORS (if not already installed)

```bash
pip install flask-cors
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
python app.py
```

The server will run on `http://localhost:5000`

### 3. Load the Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `chrome-extension` folder
5. The extension should load successfully now! ✅

## API Endpoint

The new API endpoint is available at:
- **URL**: `http://localhost:5000/api/check`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body**: `{ "url": "https://example.com" }`
- **Response**: JSON with `success`, `is_phishing`, `message`, etc.

## Optional: Add Icons Later

If you want to add icons later:
1. Open `create-icons.html` in your browser
2. Download the three icon sizes
3. Place them in `chrome-extension/icons/` folder
4. Update `manifest.json` to include the icon paths again

## Testing

1. Click the extension icon
2. Enter a URL or click "Check Current Page"
3. You should see the result (Safe ✅ or Phishing ⚠️)

