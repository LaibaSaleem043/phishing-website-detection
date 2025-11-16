# Setting Up Extension Icons

The Chrome extension needs icon files to display properly. You have two options:

## Option 1: Use the Icon Generator (Recommended)

1. Open `create-icons.html` in your web browser
2. Click each "Download" button (16x16, 48x48, 128x128)
3. Save the downloaded files as:
   - `icon16.png`
   - `icon48.png`
   - `icon128.png`
4. Place all three files in the `chrome-extension/icons/` folder

## Option 2: Create Your Own Icons

Create three PNG images with these dimensions:
- `icon16.png` - 16x16 pixels
- `icon48.png` - 48x48 pixels  
- `icon128.png` - 128x128 pixels

Use any image editor (Photoshop, GIMP, online tools, etc.) to create a shield or lock icon representing security/phishing detection.

Place all three files in the `chrome-extension/icons/` folder.

## Option 3: Use Temporary Placeholder

If you just want to test the extension quickly, you can:
1. Create a simple colored square image (any color)
2. Save it as `icon16.png`, `icon48.png`, and `icon128.png`
3. Place them in the `chrome-extension/icons/` folder

The extension will work without icons, but Chrome will show a default puzzle piece icon.

