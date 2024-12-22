import requests
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_nltk():
    """Initialize NLTK by downloading required resources."""
    required_resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'stopwords',
        'wordnet'
    ]
    
    for resource in required_resources:
        try:
            logger.info(f"Checking for NLTK resource: {resource}")
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

# Initialize NLTK resources
initialize_nltk()

# Initialize stopwords and lemmatizer after NLTK setup
original_stopwords = stopwords.words('english')
custom_stopwords = original_stopwords + ['girls', 'girl', 'another', 'problematic', 'word', 
                                       "bedwars", "wsp", "add", "welcome", "toys", "yes", "hehe"]
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess text with error handling
    """
    try:
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in custom_stopwords]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Add part-of-speech tagging with error handling
        try:
            pos_tags = nltk.pos_tag(tokens)
            pos_features = ['_'.join(tag) for word, tag in pos_tags]
        except Exception as e:
            logger.warning(f"POS tagging failed: {e}")
            pos_features = []
            
        return ' '.join(tokens + pos_features)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return ""

def load_descriptions(filename):
    """Load and preprocess descriptions from a file with error handling"""
    try:
        if not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return []
            
        with open(filename, 'r', encoding='utf-8') as file:
            descriptions = [preprocess_text(line.strip()) for line in file.readlines()]
            return [desc for desc in descriptions if desc.strip()]
    except Exception as e:
        logger.error(f"Error loading descriptions from {filename}: {e}")
        return []

# Load and preprocess descriptions
descriptions = load_descriptions('blurbs.txt')
descriptions2 = load_descriptions('blurbs2.txt')

if not descriptions or not descriptions2:
    logger.error("Failed to load descriptions. Please check your input files.")
    exit(1)

# Create labels
labels = [0] * len(descriptions)
labels2 = [1] * len(descriptions2)

# Combine descriptions and labels
combined_descriptions = descriptions + descriptions2
combined_labels = labels + labels2

# Initialize and train the model
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5)
X = vectorizer.fit_transform(combined_descriptions)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, combined_labels, test_size=0.2, random_state=42)

# Hyperparameter tuning
parameters = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0], 'fit_prior': [True, False]}
model = MultinomialNB()
grid_search = GridSearchCV(model, parameters, scoring='accuracy', cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

# Train model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, predictions))

# Cross-validation
scores = cross_val_score(best_model, X, combined_labels, cv=10)
print("Cross-validation scores:", scores)

# Print confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

print("F1 Score:", f1_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))

def predict_appropriateness(text, vectorizer, model):
    """
    Predict if a given text is appropriate or not.
    """
    try:
        preprocessed_text = preprocess_text(text)
        X_new = vectorizer.transform([preprocessed_text])
        prediction = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        return prediction[0], probabilities[0][prediction[0]]
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return 1, 0.0  # Return safe default in case of error

def process_keyword(keyword):
    """
    Process a keyword and return unique URLs
    """
    unique_urls = set()
    url_template = "https://www.roblox.com/search/users/results?keyword={keyword}&maxRows=500&startIndex={startIndex}"
    
    try:
        for startIndex in range(1, 800, 100):
            try:
                url = url_template.format(keyword=keyword, startIndex=startIndex)
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('UserSearchResults'):
                        for user in data['UserSearchResults']:
                            prediction, probability = predict_appropriateness(user.get('Blurb', ''), vectorizer, best_model)
                            if prediction == 0 and probability > 0.8:
                                full_url = f"https://roblox.com{user['UserProfilePageUrl']}"
                                unique_urls.add(full_url)
                    else:
                        logger.warning(f"No results for keyword={keyword}, startIndex={startIndex}")
                else:
                    logger.warning(f"HTTP {response.status_code} for keyword={keyword}, startIndex={startIndex}")
                    
                # Be nice to the API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing keyword={keyword}, startIndex={startIndex}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing keyword {keyword}: {e}")
    
    logger.info(f"Finished processing keyword: {keyword}")
    return unique_urls

# Load keywords and process them
try:
    with open('keywords.txt', 'r', encoding='utf-8') as file:
        keywords = [line.strip() for line in file.readlines() if line.strip()]
except Exception as e:
    logger.error(f"Error loading keywords: {e}")
    keywords = []

if not keywords:
    logger.error("No keywords loaded. Please check keywords.txt")
    exit(1)

# Process keywords with thread pool
try:
    with ThreadPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(process_keyword, keywords))
except Exception as e:
    logger.error(f"Error in thread pool execution: {e}")
    results = []

# Combine results and save
try:
    all_unique_urls = set()
    for unique_urls in results:
        all_unique_urls.update(unique_urls)

    with open('user_urls.txt', 'w', encoding='utf-8') as file:
        for url in all_unique_urls:
            file.write(url + "\n")
            
    logger.info(f"Successfully saved {len(all_unique_urls)} URLs to user_urls.txt")
    
except Exception as e:
    logger.error(f"Error saving results: {e}")

print("Processing complete. Check user_urls.txt for results and the log for any errors.")
