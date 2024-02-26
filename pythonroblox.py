import requests
import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
original_stopwords = stopwords.words('english')

# Add custom stopwords
custom_stopwords = original_stopwords + ['girls', 'girl','another', 'problematic', 'word', "bedwars", "wsp", "add", "welcome", "toys", "yes", "hehe"]
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Add part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)
    # Example of adding POS tags as features
    pos_features = ['_'.join(tag) for word, tag in pos_tags]
    return ' '.join(tokens + pos_features)

# Read descriptions from a file and preprocess
with open('blurbs.txt', 'r', encoding='utf-8') as file:
    descriptions = [preprocess_text(line.replace('\n', '')) for line in file.readlines()]

with open('blurbs2.txt', 'r', encoding='utf-8') as file:
    descriptions2 = [preprocess_text(line.replace('\n', '')) for line in file.readlines()]

# Filter out blank or whitespace-only descriptions
descriptions = [desc for desc in descriptions if desc.strip()]
descriptions2 = [desc for desc in descriptions2 if desc.strip()]

# Assuming all descriptions are inappropriate (0)
labels = [0] * len(descriptions)
labels2 = [1] * len(descriptions2)

# Combine descriptions and labels
combined_descriptions = descriptions + descriptions2
combined_labels = labels + labels2

# Preprocess data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(combined_descriptions)

# Split combined data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, combined_labels, test_size=0.2, random_state=42)

# Hyperparameter tuning
parameters = {'alpha': [0.1,  0.5,  1.0,  1.5,  2.0], 'fit_prior': [True, False]}
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
scores = cross_val_score(best_model, X, combined_labels, cv=5)
print("Cross-validation scores:", scores)

# Print confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Sort descriptions based on predictions
appropriate = []
inappropriate = []
for description, prediction in zip(combined_descriptions, predictions):
    if prediction ==  1:
        appropriate.append(description)
    else:
        inappropriate.append(description)

# Save sorted descriptions to separate files
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
    # Combine the existing data with the new data
    combined_descriptions = vectorizer.get_feature_names_out() + new_descriptions
    combined_labels = combined_labels + new_labels

    # Re-train the model with the combined data
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
    - prediction: The model's prediction (0 for inappropriate,  1 for appropriate).
    - probability: The probability of the prediction.
    """
    preprocessed_text = preprocess_text(text)
    X_new = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    return prediction[0], probabilities[0][prediction[0]]
# Define the URL template
url_template = "https://www.roblox.com/search/users/results?keyword={keyword}&maxRows=500&startIndex={startIndex}"
search_words = ["loads", "bbc", "kids", "BB(", "♠️", "fvtas", "good time", "snow bunny", "fxmboys", "dominant", "destroyed", "daddy", "insides", "studio", "BBD", "❄🐇", "𝗦𝗧𝗨𝗗𝗜𝗢", "TOP", "bottom", "dom", "fems", "ddy", "master", "Slants", "BWC", "c.m", "dumps", "femxboys", "kingky", "dommy"]

# Function to process a keyword
def process_keyword(keyword):
    unique_urls = set()
    for startIndex in range(1,   800,   100):
        url = url_template.format(keyword=keyword, startIndex=startIndex)
        response = requests.get(url)
        if response.status_code ==   200:
            data = response.json()
            # Check if 'UserSearchResults' is not None before iterating
            if data['UserSearchResults'] is not None:
                for user in data['UserSearchResults']:
                    prediction, probability = predict_appropriateness(user['Blurb'], vectorizer, best_model)
                    if prediction ==  0 and probability >  0.8:
                        full_url = "https://roblox.com"+ user['UserProfilePageUrl']
                        unique_urls.add(full_url)
            else:
                print(f"No UserSearchResults found for keyword={keyword} and startIndex={startIndex}.")
        else:
            print(f"Failed to retrieve data from Roblox API for keyword={keyword} and startIndex={startIndex}.")
    print(f"Finished processing keyword: {keyword}")
    return unique_urls
# Read keywords from a text file
with open('keywords.txt', 'r') as file:
    keywords = [line.strip() for line in file.readlines()]
# Use ThreadPoolExecutor to process keywords in parallel
with ThreadPoolExecutor(max_workers=40) as executor:
    results = executor.map(process_keyword, keywords)
# Combine results from all keywords and remove duplicates
all_unique_urls = set()
for unique_urls in results:
    all_unique_urls.update(unique_urls)
# Write all unique URLs to the file
with open('user_urls.txt', 'w') as file:
    for url in all_unique_urls:
        file.write(url + "\n")
print("All keywords have been processed and duplicates removed.")
