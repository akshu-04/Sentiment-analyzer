import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob
import joblib
import os

# Try to import TensorFlow (make it optional)
try:
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow is not available. Neural Network features will be disabled.")

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .stTextArea textarea {
            border-radius: 8px;
            padding: 10px;
        }
        .positive { color: #2ecc71; font-weight: bold; }
        .negative { color: #e74c3c; font-weight: bold; }
        .neutral { color: #3498db; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛍️ Amazon Review Sentiment Analyzer")
st.markdown("""
    Analyze sentiment of Amazon product reviews using Natural Language Processing and Machine Learning.
    This app uses both traditional ML models and neural networks for sentiment analysis.
""")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def preprocess_text(text, use_stemming=True, use_lemmatization=True):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    if use_stemming:
        words = [stemmer.stem(word) for word in words]
    if use_lemmatization:
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(r"C:\Users\Lenovo\Downloads\amazon2 (1).tsv", sep='\t')
            else:
                df = pd.read_csv(uploaded_file)
        else:
            # Try to load sample data
            sample_reviews = [
                "This product is amazing! Works perfectly.",
                "Terrible quality, would not recommend.",
                "It's okay for the price but could be better.",
                "Absolutely love it! Best purchase ever.",
                "Stopped working after 2 days. Very disappointed."
            ]
            df = pd.DataFrame({
                'verified_reviews': sample_reviews,
                'rating': [5, 1, 3, 5, 1],
                'variation': ['Black', 'White', 'Blue', 'Black', 'Red'],
                'feedback': [1, 0, 0, 1, 0]
            })
            st.warning("Using sample data as no file was uploaded")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

    # Preprocessing
    required_columns = ['verified_reviews', 'rating', 'variation', 'feedback']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default value if column missing

    df = df[required_columns].dropna()
    df['processed_text'] = df['verified_reviews'].apply(preprocess_text)
    df['review_length'] = df['verified_reviews'].apply(lambda x: len(str(x).split()))
    return df


# Model loading function
@st.cache_resource
def load_models():
    try:
        tfidf = None
        model = None

        # Try to load TF-IDF vectorizer
        if os.path.exists("tfidf_vectorizer.pkl"):
            tfidf = joblib.load("tfidf_vectorizer.pkl")

        # Try to load neural network model if TensorFlow is available
        if TENSORFLOW_AVAILABLE and os.path.exists("sentiment_nn_model.h5"):
            model = load_model("sentiment_nn_model.h5")

        return tfidf, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None


# Sidebar
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['tsv', 'csv'])
df = load_data(uploaded_file)

if df is None:
    st.error("Failed to load data. Please check your file and try again.")
    st.stop()

analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Exploratory Analysis", "Model Evaluation", "Custom Review Analysis"]
)


# Rest of your code remains the same...
# [Include all the remaining code sections here]

# Model loading function
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        model = load_model("sentiment_nn_model.h5")
        return tfidf, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None


if analysis_type == "Exploratory Analysis":
    st.header("📊 Exploratory Data Analysis")

    # Distribution plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feedback Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.countplot(x='feedback', hue='feedback', data=df, palette='coolwarm', legend=False, ax=ax1)
        ax1.set_title("Distribution of Feedback Sentiment")
        st.pyplot(fig1)

    with col2:
        st.subheader("Rating Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.countplot(x='rating', hue='rating', data=df, palette='viridis', legend=False, ax=ax2)
        ax2.set_title("Distribution of Ratings")
        st.pyplot(fig2)

    # Review length analysis
    st.subheader("Review Length Analysis")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.boxplot(x='feedback', y='review_length', hue='feedback',
                data=df, palette='coolwarm', ax=ax3, legend=False)
    ax3.set_title("Review Length by Sentiment")
    st.pyplot(fig3)

    # Word clouds
    st.subheader("Word Clouds")
    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        st.markdown("**Positive Reviews**")
        positive_text = ' '.join(df[df['feedback'] == 1]['processed_text'])
        if positive_text:
            wordcloud_pos = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
            st.image(wordcloud_pos.to_array(), width=600)
        else:
            st.warning("No positive reviews available")

    with wc_col2:
        st.markdown("**Negative Reviews**")
        negative_text = ' '.join(df[df['feedback'] == 0]['processed_text'])
        if negative_text:
            wordcloud_neg = WordCloud(width=600, height=300, background_color='black', colormap='Reds').generate(
                negative_text)
            st.image(wordcloud_neg.to_array(), width=600)
        else:
            st.warning("No negative reviews available")

    # N-grams analysis
    st.subheader("N-grams Analysis")


    def get_ngrams(text, n):
        tokens = text.split()
        return list(ngrams(tokens, n))


    df['bigrams'] = df['processed_text'].apply(lambda x: get_ngrams(x, 2))
    df['trigrams'] = df['processed_text'].apply(lambda x: get_ngrams(x, 3))

    bigram_freq = Counter([bg for row in df['bigrams'] for bg in row])
    trigram_freq = Counter([tg for row in df['trigrams'] for tg in row])

    top_bigrams = bigram_freq.most_common(10)
    top_trigrams = trigram_freq.most_common(10)

    ng_col1, ng_col2 = st.columns(2)

    with ng_col1:
        st.markdown("**Top 10 Bigrams**")
        if top_bigrams:
            bigram_df = pd.DataFrame([(" ".join(b), c) for b, c in top_bigrams],
                                     columns=["Bigram", "Count"])
            st.dataframe(bigram_df.style.background_gradient(cmap='Blues'))
        else:
            st.warning("No bigrams found")

    with ng_col2:
        st.markdown("**Top 10 Trigrams**")
        if top_trigrams:
            trigram_df = pd.DataFrame([(" ".join(t), c) for t, c in top_trigrams],
                                      columns=["Trigram", "Count"])
            st.dataframe(trigram_df.style.background_gradient(cmap='Greens'))
        else:
            st.warning("No trigrams found")

    # Additional visualizations
    st.subheader("Additional Visualizations")

    # Top 20 Words by Feedback
    st.markdown("**Top 20 Words by Feedback**")
    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    X_counts = vectorizer.fit_transform(df['processed_text'])
    word_count = pd.DataFrame(X_counts.toarray(), columns=vectorizer.get_feature_names_out())
    word_count['feedback'] = df['feedback']
    word_grouped = word_count.groupby('feedback').sum()

    fig_words, ax_words = plt.subplots(figsize=(12, 6))
    word_grouped.T.plot(kind='bar', stacked=True, colormap='Accent', ax=ax_words)
    ax_words.set_title("Top 20 Most Common Words by Feedback")
    ax_words.set_xlabel("Words")
    ax_words.set_ylabel("Frequency")
    ax_words.tick_params(axis='x', rotation=45)
    st.pyplot(fig_words)

    # TF-IDF Analysis
    st.markdown("**TF-IDF Analysis**")
    tfidf = TfidfVectorizer(max_features=10)
    X_tfidf = tfidf.fit_transform(df[df['feedback'] == 1]['processed_text'])
    top_pos = pd.Series(
        np.array(X_tfidf.mean(axis=0)).flatten(),
        index=tfidf.get_feature_names_out()
    ).sort_values(ascending=False).head(10)

    fig_tfidf, ax_tfidf = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_pos.values, y=top_pos.index,hue=top_pos.index,palette='Purples', legend=False,ax=ax_tfidf)
    ax_tfidf.set_title("Top 10 TF-IDF Terms (Positive Feedback)")
    ax_tfidf.set_xlabel("TF-IDF Score")
    ax_tfidf.set_ylabel("Word")
    st.pyplot(fig_tfidf)

    # Variation-wise Feedback Distribution
    st.markdown("**Variation-wise Feedback Distribution**")
    variation_feedback = df.groupby(['variation', 'feedback']).size().unstack().fillna(0)

    fig_var, ax_var = plt.subplots(figsize=(12, 6))
    variation_feedback.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax_var)
    ax_var.set_title('Variation-wise Feedback Distribution')
    ax_var.set_ylabel('Number of Reviews')
    ax_var.tick_params(axis='x', rotation=90)
    st.pyplot(fig_var)

    # Correlation Heatmap
    st.markdown("**Correlation Heatmap**")
    tfidf_all = TfidfVectorizer(max_features=20).fit_transform(df['processed_text'])
    tfidf_df = pd.DataFrame(tfidf_all.toarray())
    corr_df = tfidf_df.copy()
    corr_df['review_length'] = df['review_length']
    corr_matrix = corr_df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title("Correlation Matrix")
    st.pyplot(fig_corr)

elif analysis_type == "Model Evaluation":
    st.header("🤖 Model Evaluation")

    # Model selection
    model_choice = st.radio(
        "Select Model Type",
        ["Neural Network", "Support Vector Machine"],
        horizontal=True
    )

    # Prepare data
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['processed_text'])
    y = df['feedback']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_choice == "Neural Network":
        st.subheader("Neural Network Model")

        # Model architecture
        st.markdown("**Model Architecture**")
        st.code("""
        Sequential([
            Dense(128, activation='relu', input_shape=(3000,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        """)

        # Train or load model
        if st.button("Train Neural Network"):
            with st.spinner("Training in progress..."):
                nn_model = Sequential()
                nn_model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
                nn_model.add(Dropout(0.3))
                nn_model.add(Dense(64, activation='relu'))
                nn_model.add(Dropout(0.3))
                nn_model.add(Dense(1, activation='sigmoid'))

                nn_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                early_stop = EarlyStopping(monitor='val_loss', patience=3)

                history = nn_model.fit(
                    X_train.toarray(), y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_test.toarray(), y_test),
                    callbacks=[early_stop],
                    verbose=1
                )

                # Save model
                nn_model.save("sentiment_nn_model.h5")
                joblib.dump(tfidf, "tfidf_vectorizer.pkl")

                # Evaluate
                loss, test_accuracy = nn_model.evaluate(X_test.toarray(), y_test, verbose=0)
                y_probs = nn_model.predict(X_test.toarray())
                y_pred = (y_probs > 0.5).astype("int32")

                st.success("Training complete!")

                # Show metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{test_accuracy:.2%}")
                col2.metric("Precision", f"{metrics.precision_score(y_test, y_pred):.2%}")
                col3.metric("Recall", f"{metrics.recall_score(y_test, y_pred):.2%}")

                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred),
                            annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                            xticklabels=["Negative", "Positive"],
                            yticklabels=["Negative", "Positive"])
                st.pyplot(fig_cm)

                # ROC curve
                st.subheader("ROC Curve")
                fpr, tpr, _ = metrics.roc_curve(y_test, y_probs)
                roc_auc = metrics.auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                            label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

    else:  # SVM
        st.subheader("Support Vector Machine Model")

        if st.button("Train SVM Model"):
            with st.spinner("Training in progress..."):
                from sklearn.svm import SVC

                svm_model = SVC(kernel='linear', probability=True)
                svm_model.fit(X_train, y_train)

                # Save model
                joblib.dump(svm_model, "svm_model.pkl")

                # Evaluate
                y_pred = svm_model.predict(X_test)
                y_probs = svm_model.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)

                st.success("Training complete!")

                # Show metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("Precision", f"{metrics.precision_score(y_test, y_pred):.2%}")
                col3.metric("Recall", f"{metrics.recall_score(y_test, y_pred):.2%}")

                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred),
                            annot=True, fmt="d", cmap="Greens", ax=ax_cm,
                            xticklabels=["Negative", "Positive"],
                            yticklabels=["Negative", "Positive"])
                st.pyplot(fig_cm)

                # ROC curve
                st.subheader("ROC Curve")
                fpr, tpr, _ = metrics.roc_curve(y_test, y_probs)
                roc_auc = metrics.auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                            label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

else:  # Custom Review Analysis
    st.header("🔍 Custom Review Analysis")

    # Load models
    tfidf, nn_model = load_models()
    if tfidf is None or nn_model is None:
        st.warning("Please train the models first in the 'Model Evaluation' section")
        st.stop()

    # User input
    user_review = st.text_area(
        "Enter your review here:",
        placeholder="Type or paste your Amazon product review here...",
        height=150
    )

    if st.button("Analyze Sentiment"):
        if user_review.strip():
            with st.spinner("Analyzing..."):
                try:
                    # Preprocess
                    cleaned_review = preprocess_text(user_review)

                    # Vectorize
                    review_vec = tfidf.transform([cleaned_review])

                    # Predict
                    nn_prob = nn_model.predict(review_vec.toarray())[0][0]
                    nn_pred = 1 if nn_prob > 0.5 else 0

                    # TextBlob analysis
                    polarity = TextBlob(user_review).sentiment.polarity
                    subjectivity = TextBlob(user_review).sentiment.subjectivity

                    # Display results
                    st.subheader("Analysis Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Neural Network Prediction",
                            "Positive 😊" if nn_pred == 1 else "Negative 😞",
                            f"{nn_prob:.2%} confidence"
                        )

                    with col2:
                        st.metric(
                            "TextBlob Polarity",
                            f"{polarity:.2f}",
                            "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                        )

                    with col3:
                        st.metric(
                            "Subjectivity",
                            f"{subjectivity:.2f}",
                            "Opinionated" if subjectivity > 0.5 else "Factual"
                        )

                    # Sentiment visualization
                    fig_sent, ax_sent = plt.subplots(figsize=(8, 3))
                    ax_sent.barh(["Neural Network", "TextBlob"],
                                 [nn_prob, (polarity + 1) / 2],
                                 color=["#4CAF50" if nn_pred == 1 else "#F44336", "#2196F3"])
                    ax_sent.set_xlim(0, 1)
                    ax_sent.set_title("Sentiment Comparison")
                    st.pyplot(fig_sent)

                    # Show word importance
                    if nn_pred == 1:
                        st.success("""
                            **Positive Sentiment Indicators:**
                            - Words like 'love', 'great', 'excellent', 'perfect'
                            - Longer, descriptive reviews with positive adjectives
                        """)
                    else:
                        st.error("""
                            **Negative Sentiment Indicators:**
                            - Words like 'bad', 'terrible', 'disappointed'
                            - Short, frustrated reviews with negative adjectives
                        """)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please enter a review to analyze")