# Phishing Website Detection Chrome Extension

A Chrome extension that integrates with the Phishing Website Detection System backend to check if websites are safe or potentially phishing sites.

## Features

- ✅ Check current page for phishing threats
- ✅ Manually enter URLs to check
- ✅ Beautiful, modern UI
- ✅ Real-time analysis using AI-powered backend
- ✅ Configurable API endpoint

## Installation

### Step 1: Load the Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top right)
3. Click **Load unpacked**
4. Select the `chrome-extension` folder from this project

### Step 2: Start the Backend Server

Make sure the Flask backend is running:

```bash
# Navigate to the project root
cd ..

# Activate virtual environment (if using one)
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Run the Flask app
python app.py
```

The server should start on `http://localhost:5000`

### Step 3: Configure the Extension

1. Click the extension icon in Chrome toolbar
2. If your backend is running on a different URL/port, update the API URL in the footer
3. The extension will save your API URL preference

## Usage

### Check Current Page

1. Navigate to any website
2. Click the extension icon
3. Click **"Check Current Page"** button
4. View the result (Safe ✅ or Phishing ⚠️)

### Check a Specific URL

1. Click the extension icon
2. Enter a URL in the input field
3. Click **"Check URL"** button
4. View the analysis result

## Extension Structure

```
chrome-extension/
├── manifest.json       # Extension configuration
├── popup.html         # Extension popup UI
├── popup.css          # Styling for popup
├── popup.js           # Main extension logic
├── content.js         # Content script (for future enhancements)
├── icons/             # Extension icons (you'll need to add these)
└── README.md          # This file
```

## Adding Icons

The extension references icon files that you should add:

1. Create an `icons` folder in the `chrome-extension` directory
2. Add three PNG images:
   - `icon16.png` (16x16 pixels)
   - `icon48.png` (48x48 pixels)
   - `icon128.png` (128x128 pixels)

You can use any image editor or online tool to create these icons. A simple shield or lock icon would be appropriate for a phishing detection extension.

## Troubleshooting

### "Error: Server error" or Connection Issues

- Make sure the Flask backend is running (`python app.py`)
- Check that the API URL in the extension matches your server URL
- Verify the server is accessible at `http://localhost:5000` (or your configured URL)

### Extension Not Loading

- Make sure Developer mode is enabled in Chrome
- Check the browser console for errors (F12)
- Verify all files are in the correct location

### API URL Configuration

- The default API URL is `http://localhost:5000`
- You can change it in the extension popup footer
- The setting is saved automatically

## Development

To modify the extension:

1. Make changes to the files
2. Go to `chrome://extensions/`
3. Click the refresh icon on the extension card
4. Test your changes

## Future Enhancements

Potential features to add:

- [ ] Automatic page scanning on navigation
- [ ] Warning badges on suspicious links
- [ ] History of checked URLs
- [ ] Integration with browser security features
- [ ] Batch URL checking
- [ ] Export results

## License

Same as the main Phishing Website Detection System project.

