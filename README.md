# Sentiment Analysis on Social Media Data 🧠📊

This project involves building a sentiment classification system for Amazon product reviews using Machine Learning. It was developed during my internship at **AI Variant (2025)** and deployed using **Streamlit** to provide a simple user interface for business users.

## 🔍 Problem Statement

Businesses often struggle to quickly analyze and react to large volumes of customer reviews. This project aims to:
- Classify sentiments (positive/negative) from customer feedback.
- Provide automated tracking of customer sentiment trends.
- Improve product insight and customer experience strategies.

---

## 📁 Project Structure
├── data/ # Sample dataset files
├── models/ # Saved ML models (SVM, ANN)
├── sentiment_app/ # Streamlit app source code
│ └── app.py
├── notebooks/ # EDA and experimentation notebooks
├── requirements.txt
└── README.md


---

## 🚀 Features

- **Text Preprocessing**: Tokenization, stopword removal, stemming.
- **Model Training**: 
  - Support Vector Machine (SVM) — Accuracy: **91.90%**
  - Artificial Neural Network (ANN) — Accuracy: **92.95%**
- **Feature Engineering**: TF-IDF and Bag-of-Words.
- **Web App**: Built with Streamlit for interactive prediction.
- **Deployment-Ready**: Light, easy-to-integrate solution.

---

## 🧪 Tech Stack

- **Languages**: Python
- **Libraries**: NLTK, Scikit-learn, TensorFlow, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **App Framework**: Streamlit
- **Tools**: Jupyter Notebook, Git

---

## 📊 Results

- Automated sentiment tracking of customer reviews.
- Improved product feedback analysis for the business team.
- Streamlined customer experience strategy using insights.

---

## 🖥️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-streamlit.git
cd sentiment-analysis-streamlit

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the Streamlit app
streamlit run sentiment_app/app.py

🙋‍♀️ Author
Akshara A.S
📧 aksharaas041@gmail.com
🔗 LinkedIn

📄 License
This project is open-source and available under the MIT License.
