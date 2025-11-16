# Phishing Website Detection System

A comprehensive machine learning-based web application for detecting phishing websites using URL analysis. The system leverages multiple machine learning algorithms including Logistic Regression, Multinomial Naive Bayes, Word2Vec embeddings, and Transformer models (RoBERTa) to classify URLs as legitimate or phishing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [API Endpoints](#api-endpoints)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)


## Overview

This project implements a Flask-based web application that uses machine learning models to detect phishing websites. The system analyzes URL characteristics and patterns to classify websites as either legitimate ("good") or phishing ("bad"). The detection process involves:

1. **Pre-filtering**: Browser protocols and internal URLs handled before model prediction
2. **URL Preprocessing**: Tokenization, stemming, and feature extraction
3. **Feature Engineering**: Count vectorization and word embeddings
4. **Model Training**: Multiple ML algorithms trained on combined dataset (549,405 URLs)
5. **Real-time Detection**: Web interface for URL analysis

### Key Features
- ✅ **Pre-filtering**: Automatically handles browser protocols (chrome://, about:, file://, etc.) and internal URLs (localhost, internal IPs, .local domains)
- ✅ **Retrained Models**: Models trained on original dataset + safe URLs for better edge case handling
- ✅ **High Accuracy**: 96.55% accuracy with Logistic Regression, 95.80% with Multinomial Naive Bayes

## Features

- **Multiple ML Models**: Support for Logistic Regression, Multinomial Naive Bayes, Word2Vec-based models, and Transformer models
- **Enhanced Pre-filtering**: Automatic handling of browser protocols (chrome://, about:, file://) and internal URLs (localhost, 127.0.0.1, 192.168.x.x, .local domains)
- **Retrained Models**: Models trained on combined dataset (549,405 URLs) including safe URLs for better edge case handling
- **Web Interface**: User-friendly Flask web application with modern UI
- **Real-time Detection**: Instant phishing detection results
- **Data Preprocessing**: Advanced NLP techniques including tokenization and stemming
- **Model Persistence**: Pre-trained models saved using pickle for quick deployment
- **Comprehensive Analysis**: Includes data visualization and model performance metrics

## Technologies Used

### Backend & ML
- **Flask**: Web framework for the application
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas & numpy**: Data manipulation and numerical computations
- **NLTK**: Natural language processing (tokenization, stemming)
- **gensim**: Word2Vec embeddings
- **transformers**: Hugging Face Transformers (RoBERTa model)
- **xgboost**: Gradient boosting framework
- **torch**: PyTorch for deep learning models

### Frontend
- **HTML5/CSS3**: Web interface structure
- **Bootstrap 5**: Responsive UI framework
- **Tailwind CSS**: Utility-first CSS framework
- **JavaScript**: Client-side interactivity

### Data & Visualization
- **matplotlib & seaborn**: Data visualization
- **wordcloud**: Word cloud generation

## 📁 Project Structure

```
Phishing-Website-Detection-System/
│
├── app.py                              # Flask web application main file
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
│
├── Dataset/
│   └── phishing_site_urls.csv         # Training dataset (549,405 URLs: 549,346 original + 59 safe URLs)
│
├── templates/
│   └── index.html                     # Web interface HTML template
│
├── Phishing website detection system.ipynb  # Main model training notebook
├── word2vec.ipynb                     # Word2Vec and Transformer model notebook
│
├── phishing.pkl                       # Pre-trained Logistic Regression model
├── phishing_mnb.pkl                   # Pre-trained Multinomial Naive Bayes model
└── vectorizer.pkl                     # Pre-trained CountVectorizer
```

## Dataset

The project uses a dataset containing **549,405 URLs** with binary labels:

- **Good URLs**: Legitimate websites (392,983 URLs)
- **Bad URLs**: Phishing websites (156,422 URLs)

### Dataset Structure
```
URL, Label
www.youtube.com/, good
localhost:3000, good
chrome://extensions/, good
yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php, bad
...
```

### Dataset Statistics
- **Total URLs**: 549,405
- **Original URLs**: 549,346 (legitimate websites + phishing websites)
- **Safe URLs Added**: 59 (localhost, browser protocols, internal IPs, .local domains)
- **Columns**: URL, Label
- **Labels**: 'good', 'bad'
- **No missing values**

### Safe URLs Included
The dataset includes safe URLs such as:
- Localhost variations: `localhost`, `localhost:3000`, `127.0.0.1`, etc.
- Browser protocols: `chrome://extensions/`, `about:blank`, `file://`, etc.
- Internal IPs: `192.168.1.1`, `10.0.0.1`, `172.16.0.1`, etc.
- Local domains: `example.local`, `mysite.localhost`, etc.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Phishing-Website-Detection-System
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

Or in Python:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 💻 Usage

### Running the Web Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:5000` or `http://127.0.0.1:5000`

3. **Use the Application**:
   - **Enter URL**: Type or paste any URL in the input field (e.g., copy from browser address bar, email links, or manually type)
   - **Submit**: Click the "Submit" button to analyze the URL
   - **View Results**: See instant feedback showing whether the URL is safe or potentially phishing
   - The system handles various URL formats:
     - Regular URLs: `www.youtube.com`, `https://github.com`
     - Browser protocols: `chrome://extensions/`, `about:blank`
     - Internal URLs: `localhost:3000`, `127.0.0.1:8080`, `192.168.1.1`
     - Local domains: `example.local`, `mysite.localhost`

### Example Usage
```
Input: www.youtube.com/
Output: "✅ This is a safe and healthy website !!"

Input: chrome://extensions/
Output: "✅ This is a SAFE browser/system URL (chrome://, about:, file://, etc.)"

Input: localhost:3000
Output: "✅ This is a SAFE internal/local website (localhost, 127.0.0.1, .local, etc.)"

Input: yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php
Output: "⚠️ This is a Phishing website !!"
```

## 🔬 Model Training

### Main Model Training (`Phishing website detection system.ipynb`)

This notebook contains the complete pipeline:

1. **Data Loading & Exploration**
   - Load dataset
   - Check data statistics
   - Analyze label distribution

2. **Data Preprocessing**
   - URL tokenization using RegexpTokenizer
   - Stemming using SnowballStemmer
   - Text preparation

3. **Feature Extraction**
   - Count Vectorization
   - Term frequency-inverse document frequency (TF-IDF) features

4. **Model Training**
   - **Logistic Regression**: Primary model (97% accuracy)
   - **Multinomial Naive Bayes**: Alternative model

5. **Model Evaluation**
   - Classification reports
   - Confusion matrices
   - Performance metrics

6. **Model Persistence**
   - Save trained models as `.pkl` files
   - Save vectorizer for preprocessing

### Advanced Models (`word2vec.ipynb`)

This notebook explores advanced techniques:

1. **Word2Vec Embeddings**
   - Train Word2Vec model on URL tokens
   - Create URL embeddings by averaging token vectors
   - Use with traditional ML algorithms (XGBoost)

2. **Transformer Models**
   - Fine-tune RoBERTa-base model
   - Hugging Face Transformers integration
   - Advanced deep learning approach

### Training Your Own Model

1. **Open Jupyter Notebook**:
```bash
jupyter notebook "Phishing website detection system.ipynb"
```

2. **Run all cells** to train models from scratch

3. **Models will be saved** as:
   - `phishing.pkl` (Logistic Regression)
   - `phishing_mnb.pkl` (Multinomial Naive Bayes)
   - `vectorizer.pkl` (Feature vectorizer)

## 🌐 Web Application

### How Users Interact with the Application

Users can access the phishing detection system through a **web browser interface** where they:

1. **Enter URL in the Input Field**: Users paste or type any URL they want to check in the web form
2. **Submit for Analysis**: Click the "Submit" button to analyze the URL
3. **View Results**: Receive instant feedback indicating whether the URL is safe or potentially phishing

### Application Flow

1. **User Input**: User enters URL in the web browser extension/interface
   - User can paste any URL (http://, https://, localhost, chrome://, etc.)
   - URL can be copied from browser address bar or manually typed
   
2. **Pre-filtering** (Enhanced):
   - Check if browser protocol (chrome://, about:, file://, etc.) → Mark as safe immediately
   - Check if internal URL (localhost, 127.0.0.1, 192.168.x.x, .local, etc.) → Mark as safe immediately
   
3. **URL Preprocessing** (if not pre-filtered):
   - Remove protocol (http://, https://)
   - Remove www. prefix
   - Tokenize and stem URL text
   - Extract features
   
4. **Model Prediction**: Use pre-trained retrained model for classification
   - Model analyzes URL features using Logistic Regression
   - Returns prediction: 'good' (legitimate) or 'bad' (phishing)
   
5. **Result Display**: Show prediction with user-friendly message
   - ✅ Safe website: "This is a safe and healthy website !!"
   - ✅ Browser/system URL: "This is a SAFE browser/system URL..."
   - ✅ Internal/local URL: "This is a SAFE internal/local website..."
   - ⚠️ Phishing: "This is a Phishing website !!"

### Enhanced Features (`app.py`)

The application now includes intelligent pre-filtering:

```python
# Pre-filtering functions
def is_browser_protocol(url):
    """Detects browser protocols: chrome://, about:, file://, etc."""
    # Returns True for safe browser protocols

def is_internal_url(url):
    """Detects internal/local URLs: localhost, internal IPs, .local domains"""
    # Returns True for safe internal URLs

# Application logic
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        url = request.form['url']
        
        # Pre-filter browser protocols and internal URLs
        if is_browser_protocol(url):
            return "SAFE browser/system URL"
        if is_internal_url(url):
            return "SAFE internal/local website"
        
        # Use ML model for other URLs
        cleaned_url = re.sub(r'^https?://(www\.)?', '', url)
        predict = model.predict(vector.transform([cleaned_url]))[0]
        # Return prediction result
    return render_template("index.html")
```

### Code Structure (`app.py`)

```python
# Load pre-trained retrained models
vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("phishing.pkl", 'rb'))

# Enhanced route handler with pre-filtering
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        url = request.form['url']
        
        # Pre-filtering (handles edge cases immediately)
        if is_browser_protocol(url):
            return "SAFE browser/system URL"
        if is_internal_url(url):
            return "SAFE internal/local website"
        
        # ML model prediction for remaining URLs
        cleaned_url = re.sub(r'^https?://(www\.)?', '', url)
        predict = model.predict(vector.transform([cleaned_url]))[0]
        # Return prediction result
    return render_template("index.html")
```

## 🔌 API Endpoints

### POST `/`
- **Purpose**: Analyze URL for phishing detection
- **Method**: POST
- **Parameters**:
  - `url` (form data): URL to analyze
- **Response**: HTML page with prediction result

### GET `/`
- **Purpose**: Display the main form
- **Method**: GET
- **Response**: HTML form page

## 📈 Model Performance

### Retrained Models (Current)

The models have been retrained on a combined dataset (549,405 URLs) that includes:
- Original dataset: 549,346 URLs (392,924 good + 156,422 bad)
- Safe URLs: 59 URLs (localhost, browser protocols, internal IPs, .local domains)

### Logistic Regression Model (Primary)
- **Test Accuracy**: 96.55%
- **Train Accuracy**: 97.99%
- **Precision (Bad)**: 0.97
- **Recall (Bad)**: 0.91
- **F1-Score (Bad)**: 0.94
- **Precision (Good)**: 0.96
- **Recall (Good)**: 0.99
- **F1-Score (Good)**: 0.98
- **Features**: 350,840
- **Status**: ✅ Retrained with combined dataset

### Multinomial Naive Bayes Model (Alternative)
- **Test Accuracy**: 95.80%
- **Train Accuracy**: 97.42%
- **Precision (Bad)**: 0.94
- **Recall (Bad)**: 0.92
- **F1-Score (Bad)**: 0.93
- **Precision (Good)**: 0.97
- **Recall (Good)**: 0.97
- **F1-Score (Good)**: 0.97
- **Features**: 350,840
- **Status**: ✅ Retrained with combined dataset

### Confusion Matrix Results
```
                    Predicted: Bad    Predicted: Good
Actual: Bad              28,418          893
Actual: Good             3,141          77,418
```

### Classification Report
```
              precision    recall  f1-score   support

         Bad       0.91      0.97      0.94     29,311
        Good       0.99      0.96      0.98     80,559

    accuracy                           0.97    109,870
   macro avg       0.95      0.97      0.96    109,870
weighted avg       0.97      0.97      0.97    109,870
```

## 🔄 Workflow

```
┌─────────────┐
│   User URL  │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Pre-filtering       │
│  • Browser protocols │ (chrome://, about:, file://)
│  • Internal URLs     │ (localhost, 127.0.0.1, .local)
└──────┬───────────────┘
       │
       │ (If not pre-filtered)
       ▼
┌──────────────────┐
│ URL Preprocessing│ (Remove protocol, www, tokenize, stem)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Feature Extraction│ (Count Vectorization - 350,840 features)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Model Prediction │ (Retrained Logistic Regression)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Display Result  │ (Phishing or Legitimate)
└──────────────────┘
```

## 🎨 Features of the Web Interface

- **Modern UI**: Clean, responsive design using Bootstrap and Tailwind CSS
- **Real-time Detection**: Instant results upon form submission
- **User-friendly Interface**: Simple input field where users can enter any URL
- **Multiple URL Format Support**: Handles regular URLs, browser protocols, internal IPs, and local domains
- **Clear Result Display**: Visual feedback with emojis and clear messages (✅ Safe, ⚠️ Phishing)
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Easy to Use**: Users simply paste or type URL in the input field and click Submit

## 🔮 Future Enhancements

1. **Real-time URL Scanning**: Implement background URL verification
2. **API Integration**: RESTful API for programmatic access
3. **Database Integration**: Store detection history
4. **Enhanced Models**: Integrate transformer models in production
5. **Batch Processing**: Support for analyzing multiple URLs at once
6. **URL Feature Extraction**: Additional features like domain age, SSL certificates
7. **Dashboard**: Analytics dashboard for detection statistics
8. **Email Notifications**: Alert users about detected phishing sites
9. **Model Retraining**: Automated model retraining with new data
10. **Docker Support**: Containerize the application for easy deployment

## 🧪 Testing

### Test URLs

**Legitimate URLs**:
- `www.youtube.com/` → ✅ Safe and healthy website
- `youtube.com/watch?v=qI0TQJI3vdU` → ✅ Safe and healthy website
- `www.retailhellunderground.com/` → ✅ Safe and healthy website
- `restorevisioncenters.com/html/technology.html` → ✅ Safe and healthy website

**Browser Protocols** (Pre-filtered):
- `chrome://extensions/` → ✅ SAFE browser/system URL
- `about:blank` → ✅ SAFE browser/system URL
- `file:///path/to/file` → ✅ SAFE browser/system URL
- `edge://settings/` → ✅ SAFE browser/system URL

**Internal URLs** (Pre-filtered):
- `localhost:3000` → ✅ SAFE internal/local website
- `127.0.0.1:8080` → ✅ SAFE internal/local website
- `192.168.1.1` → ✅ SAFE internal/local website
- `example.local` → ✅ SAFE internal/local website

**Phishing URLs**:
- `yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php` → ⚠️ Phishing website
- `fazan-pacir.rs/temp/libraries/ipad` → ⚠️ Phishing website
- `www.tubemoviez.exe` → ⚠️ Phishing website
- `svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt` → ⚠️ Phishing website



## Notes

- The models are pre-trained and retrained with combined dataset (549,405 URLs), saved as `.pkl` files for quick deployment
- Models contain knowledge from both original dataset and safe URLs (localhost, browser protocols, internal IPs, etc.)
- Make sure all model files (`phishing.pkl`, `phishing_mnb.pkl`, `vectorizer.pkl`) are present before running the application
- The application includes enhanced pre-filtering for browser protocols and internal URLs before model prediction
- For production deployment, set `debug=False` in `app.py`
- Consider using a production WSGI server like Gunicorn for deployment

## Model Training & Retraining

### Original Training
The models were initially trained on 549,346 URLs using the Jupyter notebook (`Phishing website detection system.ipynb`).

### Retraining Process
The models were retrained with additional safe URLs:
- **Script**: `retrain_model.py`
- **Safe URLs Dataset**: `safe_urls_dataset.csv` (59 URLs)
- **Process**: 
  1. Loads original dataset (549,346 URLs)
  2. Adds safe URLs (59 URLs)
  3. Saves combined dataset to `Dataset/phishing_site_urls.csv`
  4. Retrains both models (Logistic Regression & Multinomial Naive Bayes)
  5. Saves updated models

### Current Models
- **Training Data**: 549,405 URLs (549,346 original + 59 safe URLs)
- **Features**: 350,840 features
- **Status**: Retrained with combined dataset
- **Accuracy**: 96.55% (Logistic Regression), 95.80% (Multinomial Naive Bayes)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: 
   - Solution: Install all dependencies using `pip install -r requirements.txt`

2. **Model files not found**:
   - Solution: Ensure `phishing.pkl`, `vectorizer.pkl` are in the project root directory
   - If missing, run the training notebook to generate them

3. **NLTK data missing**:
   - Solution: Download required NLTK data (see Installation section)

4. **Port already in use**:
   - Solution: Change the port in `app.py`: `app.run(debug=True, port=5001)`


## 👥 Authors

- **Project Maintainer** - Laiba Saleem

## 🙏 Acknowledgments

- Dataset contributors
- scikit-learn, NLTK, and Hugging Face communities
- Flask framework developers

#   p h i s h i n g - w e b s i t e - d e t e c t i o n  
 