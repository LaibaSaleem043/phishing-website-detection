from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("phishing.pkl", 'rb'))




def is_browser_protocol(url):
    """Check if URL is a browser protocol (chrome://, about:, file://, etc.)"""
    browser_protocols = [
        'chrome://',
        'chrome-extension://',
        'moz-extension://',
        'about:',
        'file://',
        'data:',
        'javascript:',
        'edge://',
        'brave://'
    ]
    url_lower = url.lower().strip()
    return any(url_lower.startswith(proto) for proto in browser_protocols)


def is_internal_url(url):
    """Check if URL is localhost, internal IP, or local domain"""
    cleaned = re.sub(r'^https?://(www\.)?', '', url).strip().lower()
    cleaned = re.sub(r'^ftp://', '', cleaned).strip().lower()
    
    # Remove port numbers
    cleaned = re.sub(r':\d+$', '', cleaned).strip()
    
    # Extract just the hostname/IP
    if '/' in cleaned:
        cleaned = cleaned.split('/')[0]
    
    # Local development URLs
    internal_hosts = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "localhost.localdomain"
    ]

    if cleaned in internal_hosts:
        return True

    # Private IP ranges (RFC 1918)
    # 10.0.0.0/8
    if cleaned.startswith("10."):
        return True
    
    # 172.16.0.0/12
    if cleaned.startswith("172."):
        parts = cleaned.split('.')
        if len(parts) >= 2:
            try:
                second = int(parts[1])
                if 16 <= second <= 31:
                    return True
            except:
                pass
    
    # 192.168.0.0/16
    if cleaned.startswith("192.168."):
        return True
    
    # 127.0.0.0/8 (loopback)
    if cleaned.startswith("127."):
        return True
    
    # 169.254.0.0/16 (link-local)
    if cleaned.startswith("169.254."):
        return True
    
    # .local domains
    if cleaned.endswith(".local") or ".local." in cleaned:
        return True
    
    # .localhost domains
    if cleaned.endswith(".localhost") or ".localhost." in cleaned:
        return True

    return False



@app.route("/api/check", methods=['POST'])
def check_url_api():
    """API endpoint for Chrome extension - returns JSON response"""
    try:
        # Get URL from JSON or form data
        if request.is_json:
            data = request.get_json()
            url = data.get('url', '')
        else:
            url = request.form.get('url', '') or request.args.get('url', '')
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required',
                'message': 'Please provide a URL to check'
            }), 400
        

        # CHECK BROWSER PROTOCOLS FIRST (chrome://, about:, file://, etc.)
        if is_browser_protocol(url):
            result = {
                'success': True,
                'is_phishing': False,
                'message': 'This is healthy and good website !!',
                'url': url,
                'cleaned_url': url
            }
            return result 

        # CHECK INTERNAL/LOCALHOST URLs
        if is_internal_url(url):
            result = {
                'success': True,
                'is_phishing': False,
                'message': 'This is healthy and good website !!',
                'url': url,
                'cleaned_url': url
            }
            return result


        cleaned_url = re.sub(r'^https?://(www\.)?', '', url).strip().lower()
        cleaned_url = re.sub(r'^ftp://', '', cleaned_url).strip()

        # If cleaned URL is empty or too short, might be invalid
        if not cleaned_url or len(cleaned_url) < 3:
            result = {
                'success': True,
                'is_phishing': True,
                'message': 'This is a Phishing website !!',
                'url': url,
                'cleaned_url': cleaned_url
            }

            return result
        
        # Clean the URL
        
        # Make prediction
        prediction = model.predict(vector.transform([cleaned_url]))[0]
        
        # Format response
        if prediction == 'bad':
            result = {
                'success': True,
                'is_phishing': True,
                'message': 'This is a Phishing website !!',
                'url': url,
                'cleaned_url': cleaned_url
            }
        elif prediction == 'good':
            result = {
                'success': True,
                'is_phishing': False,
                'message': 'This is healthy and good website !!',
                'url': url,
                'cleaned_url': cleaned_url
            }
        else:
            result = {
                'success': False,
                'error': 'Unknown prediction result',
                'message': 'Something went wrong !!',
                'url': url
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while processing the request'
        }), 500



if __name__=="__main__":
    app.run(debug=True)