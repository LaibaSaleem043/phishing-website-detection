🛡️ Phishing Website Detection System

A Machine Learning–based system to detect phishing websites using URL features, ML models (Multinomial Naive Bayes, Word2Vec), and Flask for web deployment.

Features

Detect phishing URLs in real-time

ML models trained on URL-based features

Multinomial Naive Bayes classifier

Word2Vec feature extraction

Web interface built with Flask

Simple and fast predictions

📂 Project Structure
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

Machine Learning Models Used
✔️ Multinomial Naive Bayes

Used for text-based classification of URLs.

✔️ Word2Vec

Used to convert URL tokens into vector representations.

🔧 Installation & Setup
1️⃣ Clone the Repository
git clone <your_repo_url>
cd Phishing-Website-Detection-System

2️⃣ Create Virtual Environment
python -m venv venv

Activate Environment

Windows PowerShell

venv\Scripts\Activate.ps1


If PowerShell blocks activation:

Set-ExecutionPolicy Unrestricted -Scope Process

3️⃣ Install Dependencies
pip install -r requirements.txt


If NLTK errors appear, install required packages:

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

▶️ Running the Application

Start the Flask server:

python app.py


You should see:

 * Running on http://127.0.0.1:5000


Open your browser and visit:

👉 http://127.0.0.1:5000

or
👉 http://localhost:5000

Enter any URL to test phishing detection.

🧪 Testing the Model

Use URLs like:

Phishing Example URLs
http://paypa1-login-secure.com
http://update-banking-info-support.net
http://facebook-security-check-verify.gq
http://appleid-login-verify-account.ga

Legitimate URLs
https://www.google.com
https://www.microsoft.com
https://www.github.com

🛠️ Tech Stack

Python

Flask

Scikit-learn

NLTK

Word2Vec

HTML/CSS

📌 Future Improvements

Chrome extension version

Model endpoint for API integration

Add more phishing datasets

Use deep learning (LSTM/CNN)

🤝 Contributing

Contributions are welcome!
Please open issues or submit pull requests.
