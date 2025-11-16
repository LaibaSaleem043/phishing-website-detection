```markdown
# Phishing Website Detection System 🛡️

A Machine Learning–based system to detect phishing websites using URL features, ML models (Multinomial Naive Bayes, Word2Vec), and Flask for a simple web interface.

---

## Features

- Real-time phishing URL detection via a web UI
- ML models trained on URL-based features
  - Multinomial Naive Bayes classifier
  - Word2Vec token-based feature extraction
- Lightweight Flask web app for quick local testing
- Fast, text-based predictions

---

## Project Structure

Phishing-Website-Detection-System/

Phishing-Website-Detection-System/
│
├── app.py                         # Flask web application
├── phishing.pkl                   # Trained model
├── phishing_mnb.pkl               # MNB classifier
├── vectorizer.pkl                 # TF-IDF vectorizer
├── Dataset/                       # Dataset files
├── templates/
│   └── index.html                 # Web UI
├── venv/                          # Virtual environment
├── requirements.txt               # Dependencies
└── README.md



---
```
## Prerequisites

- Python 3.8+
- pip

(Optional) Create and use a virtual environment to avoid dependency conflicts.

---

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/LaibaSaleem043/phishing-website-detection.git
cd phishing-website-detection
```

2. Create a virtual environment
```bash
python -m venv venv
```

3. Activate the virtual environment

- macOS / Linux:
```bash
source venv/bin/activate
```

- Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

- Windows (Command Prompt):
```cmd
venv\Scripts\activate.bat
```

If PowerShell blocks activation you can run:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\Activate.ps1
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. (If NLTK errors appear) download required NLTK packages:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Running the Application

Start the Flask server:
```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

Open your browser and load unpacked extension:
- chrome-extension

Enter any URL to test phishing detection.

---

## Example URLs for Testing

Phishing (examples)
- http://paypa1-login-secure.com
- http://update-banking-info-support.net
- http://facebook-security-check-verify.gq
- http://appleid-login-verify-account.ga

Legitimate
- https://www.google.com
- https://www.microsoft.com
- https://www.github.com

---

## Models & Files

- phishing_mnb.pkl — Multinomial Naive Bayes classifier trained on URL features
- vectorizer.pkl — TF-IDF / token vectorizer used for feature extraction
- phishing.pkl — (Optional) combined or alternate trained model
- Dataset/ — original or processed datasets used for training

If you retrain models locally, make sure to save them with the same filenames (or update app.py to point to the new filenames).

---

## Tech Stack

- Python
- Flask
- scikit-learn
- NLTK
- Gensim / Word2Vec (if used)
- HTML/CSS for the simple front-end

---

## Future Improvements

- Chrome extension for in-browser detection
- Expose model as an API endpoint for integrations
- Expand and curate more phishing datasets
- Replace or augment with deep learning models (LSTM/CNN) for improved accuracy
- Add tests and CI

---

## Contributing

Contributions are welcome! Please open issues to report bugs or request features, and submit pull requests for proposed changes.

```
