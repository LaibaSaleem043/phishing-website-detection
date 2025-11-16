// Default API URL
const DEFAULT_API_URL = 'http://localhost:5000';

// DOM elements
const urlInput = document.getElementById('urlInput');
const checkCurrentBtn = document.getElementById('checkCurrentBtn');
const checkUrlBtn = document.getElementById('checkUrlBtn');
const resultDiv = document.getElementById('result');
const resultIcon = document.getElementById('resultIcon');
const resultText = document.getElementById('resultText');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const errorText = document.getElementById('errorText');
const apiUrlInput = document.getElementById('apiUrl');

// Load saved API URL
chrome.storage.sync.get(['apiUrl'], (data) => {
  if (data.apiUrl) {
    apiUrlInput.value = data.apiUrl;
  } else {
    apiUrlInput.value = DEFAULT_API_URL;
  }
});

// Save API URL when changed
apiUrlInput.addEventListener('change', () => {
  chrome.storage.sync.set({ apiUrl: apiUrlInput.value });
});

// Check current page button
checkCurrentBtn.addEventListener('click', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab.url) {
      urlInput.value = tab.url;
      await checkUrl(tab.url);
    } else {
      showError('Unable to get current page URL');
    }
  } catch (error) {
    showError('Error accessing current tab: ' + error.message);
  }
});

// Check URL button
checkUrlBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) {
    showError('Please enter a URL');
    return;
  }
  await checkUrl(url);
});

// Enter key support
urlInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    checkUrlBtn.click();
  }
});

// Main function to check URL
async function checkUrl(url) {
  // Hide previous results
  hideAll();
  showLoading();

  try {
    // Get API URL
    const apiUrl = apiUrlInput.value.trim() || DEFAULT_API_URL;
    
    // Normalize URL
    let normalizedUrl = url;
    if (!normalizedUrl.startsWith('http://') && !normalizedUrl.startsWith('https://')) {
      normalizedUrl = 'https://' + normalizedUrl;
    }

    // Make API request to JSON endpoint
    const response = await fetch(`${apiUrl}/api/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: normalizedUrl })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || errorData.message || `Server error: ${response.status}`);
    }

    // Parse JSON response
    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || data.message || 'Unknown error occurred');
    }

    // Display result
    hideLoading();
    showResult(data.message, data.is_phishing);

  } catch (error) {
    hideLoading();
    showError(`Error: ${error.message}. Make sure the backend server is running at ${apiUrlInput.value || DEFAULT_API_URL}`);
  }
}

// Show loading state
function showLoading() {
  loadingDiv.classList.remove('hidden');
}

// Hide loading state
function hideLoading() {
  loadingDiv.classList.add('hidden');
}

// Show result
function showResult(message, isPhishing) {
  resultDiv.classList.remove('hidden');
  
  if (isPhishing === true) {
    resultDiv.className = 'result phishing';
    resultIcon.textContent = '⚠️';
    resultText.textContent = message;
  } else if (isPhishing === false) {
    resultDiv.className = 'result safe';
    resultIcon.textContent = '✅';
    resultText.textContent = message;
  } else {
    resultDiv.className = 'result error-state';
    resultIcon.textContent = '❓';
    resultText.textContent = message || 'Unknown result';
  }
}

// Show error
function showError(message) {
  errorDiv.classList.remove('hidden');
  errorText.textContent = message;
}

// Hide all result divs
function hideAll() {
  resultDiv.classList.add('hidden');
  errorDiv.classList.add('hidden');
  loadingDiv.classList.add('hidden');
}

