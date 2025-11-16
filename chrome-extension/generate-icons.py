"""
Simple script to generate placeholder icons for the Chrome extension.
Run this script to create the required icon files.
"""

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow is not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageDraw

import os

def create_icon(size, filename):
    """Create a simple icon with a shield shape"""
    # Create a new image with gradient background
    img = Image.new('RGB', (size, size), color='#667eea')
    draw = ImageDraw.Draw(img)
    
    # Draw gradient effect (simple version)
    for i in range(size):
        ratio = i / size
        r = int(102 + (118 - 102) * ratio)  # 667eea to 764ba2
        g = int(126 + (75 - 126) * ratio)
        b = int(234 + (162 - 234) * ratio)
        draw.rectangle([(0, i), (size, i+1)], fill=(r, g, b))
    
    # Draw shield shape
    center_x, center_y = size // 2, size // 2
    shield_size = int(size * 0.6)
    
    # Shield points
    points = [
        (center_x, center_y - shield_size // 3),
        (center_x + shield_size // 4, center_y - shield_size // 10),
        (center_x + shield_size // 4, center_y + shield_size // 5),
        (center_x, center_y + shield_size // 2),
        (center_x - shield_size // 4, center_y + shield_size // 5),
        (center_x - shield_size // 4, center_y - shield_size // 10),
    ]
    
    draw.polygon(points, fill='white')
    
    # Draw checkmark
    line_width = max(2, size // 16)
    check_points = [
        (center_x - shield_size // 6, center_y),
        (center_x - shield_size // 20, center_y + shield_size // 10),
        (center_x + shield_size // 6, center_y - shield_size // 10),
    ]
    draw.line([check_points[0], check_points[1], check_points[2]], 
              fill='#667eea', width=line_width)
    
    # Save the image
    img.save(filename, 'PNG')
    print(f"Created {filename} ({size}x{size})")

def main():
    # Create icons directory if it doesn't exist
    icons_dir = 'icons'
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
        print(f"Created {icons_dir} directory")
    
    # Generate icons
    sizes = [16, 48, 128]
    for size in sizes:
        filename = os.path.join(icons_dir, f'icon{size}.png')
        create_icon(size, filename)
    
    print("\n✅ All icons generated successfully!")
    print("You can now load the Chrome extension.")

if __name__ == '__main__':
    main()


