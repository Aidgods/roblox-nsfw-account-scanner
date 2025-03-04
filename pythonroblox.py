import requests
import time
import json
import requests.exceptions
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.downloader import Downloader
from pathlib import Path
import logging
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import List
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
original_stopwords = stopwords.words('english')


custom_stopwords = original_stopwords + ['girls', 'girl','another', 'problematic', 'word', "bedwars", "wsp", "add", "welcome", "toys", "yes", "hehe"]
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    pos_tags = nltk.pos_tag(tokens)

    pos_features = ['_'.join(tag) for word, tag in pos_tags]
    return ' '.join(tokens + pos_features)


with open('blurbs.txt', 'r', encoding='utf-8') as file:
    descriptions = [preprocess_text(line.replace('\n', '')) for line in file.readlines()]

with open('blurbs2.txt', 'r', encoding='utf-8') as file:
    descriptions2 = [preprocess_text(line.replace('\n', '')) for line in file.readlines()]


descriptions = [desc for desc in descriptions if desc.strip()]
descriptions2 = [desc for desc in descriptions2 if desc.strip()]


labels = [0] * len(descriptions)
labels2 = [1] * len(descriptions2)


combined_descriptions = descriptions + descriptions2
combined_labels = labels + labels2


vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.7, min_df=2, analyzer='char_wb')
X = vectorizer.fit_transform(combined_descriptions)


X_train, X_test, y_train, y_test = train_test_split(X, combined_labels, test_size=0.2, random_state=42)

# Hyperparameter tuning
parameters = {
    'model__C': [0.1, 1.0, 10.0, 100.0],  
    'model__penalty': ['l2'],       # Regularization type
    'model__solver': ['lbfgs'],     
    'model__max_iter': [100, 200, 500, 1000]  
}

model = LogisticRegression(class_weight={0: 4, 1: 1}, random_state=42)
smote = SMOTE(sampling_strategy="auto", random_state=42)


pipeline = ImbPipeline([
    ('smote', smote),  
    ('model', model)  
])
grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)



print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)


predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)


print("Accuracy:", accuracy_score(y_test, predictions))


scores = cross_val_score(best_model, X, combined_labels, cv=10)
print("Cross-validation scores:", scores)


print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

print("F1 Score:", f1_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))


appropriate = []
inappropriate = []
for description, prediction in zip(combined_descriptions, predictions):
    if prediction == 1:
        appropriate.append(description)
    else:
        inappropriate.append(description)


with open('appropriate.txt', 'w', encoding='utf-8') as file:
    for desc in appropriate:
        file.write(desc + "\n")

with open('inappropriate.txt', 'w', encoding='utf-8') as file:
    for desc in inappropriate:
        file.write(desc + "\n")

print("Descriptions have been sorted into 'appropriate.txt' and 'inappropriate.txt'.")

def update_model_with_new_data(new_descriptions, new_labels, vectorizer, model):
    """
    Update the existing model with new data.

    Parameters:
    - new_descriptions: List of new text descriptions.
    - new_labels: Corresponding labels for the new descriptions.
    - vectorizer: The TfidfVectorizer used during training.
    - model: The existing trained MultinomialNB model.
    """

    combined_descriptions = vectorizer.get_feature_names_out() + new_descriptions
    combined_labels = combined_labels + new_labels


    X = vectorizer.fit_transform(combined_descriptions)
    model.fit(X, combined_labels)

def predict_appropriateness(text, vectorizer, model):
    """
    Predict if a given text is appropriate or not.

    Parameters:
    - text: The text to be evaluated.
    - vectorizer: The TfidfVectorizer used during training.
    - model: The trained MultinomialNB model.

    Returns:
    - prediction: The model's prediction (0 for inappropriate, 1 for appropriate).
    - probability: The probability of the prediction.
    """
    preprocessed_text = preprocess_text(text)
    X_new = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    return prediction[0], probabilities[0][prediction[0]]
# Define the URL template
# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

def process_keyword(keyword, max_users=10, max_pages=1):
    """
    Process a keyword and return unique URLs using the new Roblox API format.
    
    Args:
        keyword (str): The keyword to search for.
        max_users (int): Maximum number of users to process per keyword.
        max_pages (int): Maximum number of pages to fetch per keyword.
    """
    unique_urls = set()
    url_template = "https://users.roproxy.com/v1/users/search?keyword=!{keyword}&limit=100"
    
    try:
        next_page_cursor = None
        processed_users = 0
        processed_pages = 0
        
        while processed_pages < max_pages and processed_users < max_users:
            try:
                # Construct the URL with the cursor for pagination
                url = url_template.format(keyword=keyword)
                if next_page_cursor:
                    url += f"&cursor={next_page_cursor}"
                
                # Add a delay to avoid rate limiting
                time.sleep(5)  # Increase the delay to 5 seconds
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process user data
                    for user in data.get('data', []):
                        if processed_users >= max_users:
                            break  # Stop if we've processed enough users
                        
                        user_id = user.get('id')
                        if user_id:
                            # Fetch user details using the user ID
                            user_details_url = f"https://users.roproxy.com/v1/users/{user_id}"
                            time.sleep(3)  # Add a delay between user detail requests
                            user_details_response = requests.get(user_details_url)
                            
                            if user_details_response.status_code == 200:
                                user_details = user_details_response.json()
                                blurb = user_details.get('description', '')
                                
                                # Predict appropriateness
                                prediction, probability = predict_appropriateness(blurb, vectorizer, best_model)
                                if prediction == 0 and probability > 0.8:
                                    full_url = f"https://www.roblox.com/users/{user_id}/profile"
                                    unique_urls.add(full_url)
                                    processed_users += 1  # Increment the processed user count
                            else:
                                logger.warning(f"Failed to fetch user details for user ID {user_id}: HTTP {user_details_response.status_code}")
                    
                    # Check for pagination
                    next_page_cursor = data.get('nextPageCursor')
                    if not next_page_cursor:
                        break
                    
                    processed_pages += 1  # Increment the processed page count
                else:
                    logger.warning(f"HTTP {response.status_code} for keyword={keyword}")
                    break
                
            except Exception as e:
                logger.error(f"Error processing keyword={keyword}: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error processing keyword {keyword}: {e}")
    
    logger.info(f"Finished processing keyword: {keyword}. Processed {processed_users} users.")
    return unique_urls

# Example usage
with open('keywords.txt', 'r') as file:
    keywords = [line.strip() for line in file.readlines()]

# Process keywords with limits
all_unique_urls = set()
for keyword in keywords:
    unique_urls = process_keyword(keyword, max_users=100, max_pages=1)  # Adjust limits as needed
    all_unique_urls.update(unique_urls)

# Write results to file
with open('user_urls.txt', 'w') as file:
    for url in all_unique_urls:
        file.write(url + "\n")

logger.info("All keywords have been processed and duplicates removed.")
