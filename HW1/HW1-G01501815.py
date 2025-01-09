import numpy as np
import pandas as pd
import nltk
import re
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize


file = "Train_data.txt"
stop_words = set(stopwords.words('english'))
unwanted_words = {"br"}

def preprocess_text(text):
    # Remove HTML tags like <br> and other unwanted characters
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", '', text)

    word_tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in word_tokens:
        if word not in stop_words and word not in unwanted_words:
            lemmatized_word = lemmatizer.lemmatize(word)
            lemmas.append(lemmatized_word)
    return ' '.join(lemmas)


def process_reviews_from_file(file):
    preprocessed_data = []

    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # used label for +1 and -1 ratings
                label = int(line[0:2].strip())

                # Replace the label -1 with 2 ( This is done because bincount doesn't accecpts negative characters )
                if label == -1:
                    label = 2

                review = line[2:].strip()

                # Removing "EOF" at the end of each para
                if review.endswith("EOF"):
                    review = review[:-3].strip()
                review_text = preprocess_text(review)

                # Append the result in the desired format
                preprocessed_data.append([review_text, label])
    return preprocessed_data

data = process_reviews_from_file(file)
labels = [item[1] for item in data]
print("Negative labels:", [label for label in labels if label < 0])

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidf_vectorizer.fit_transform([df[0] for df in data]).toarray()
Y_train = [df[1] for df in data]

class KNN:
    def __init__(self, k):
        self.Y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    # using np for all operations and everything is done in whole matrix level
    # this optimization improve the performance more than 10x compared to using for loops
    def cosine_similarity(self, x, y):

        norms_vector = np.linalg.norm(x, axis=1)
        norms_vectors = np.linalg.norm(y, axis=1)

        # Handle cases where the norms are zero or very close to zero
        norms_vector[norms_vector == 0] = 1e-10
        norms_vectors[norms_vectors == 0] = 1e-10

        similarity_matrix = np.dot(y, x.T)

        return similarity_matrix / np.outer(norms_vectors, norms_vector)

    def predict(self, x):
        cosine_similarity = self.cosine_similarity(x, self.X_train)

        # Find k-nearest neighbors efficiently using argsort
        knn_indices = np.argsort(cosine_similarity.T, axis=1)[:, -self.k:]

        # Retrieve the labels for the k-nearest neighbors
        knn_labels = self.Y_train[knn_indices]

        # Predict the labels by finding the most common label among the neighbors
# Ensure all labels are non-negative integers
        prediction = np.array([np.argmax(np.bincount(np.clip(labels.astype(int), 0, None))) for labels in knn_labels])

        return prediction


from sklearn.preprocessing import StandardScaler

def cross_validation(predictions, y_test):
    true_positive = true_negative = false_positive = false_negative = 0
    for t_label, p_label in zip(y_test, predictions):
        if t_label == 1 and p_label == 1:
            true_positive += 1
        elif t_label == 2 and p_label == 2:
            true_negative += 1
        elif t_label == 2 and p_label == 1:
            false_positive += 1
        elif t_label == 1 and p_label == 2:
            false_negative += 1
    return [true_positive, true_negative, false_positive, false_negative]

def k_fold_cross_validation(x, y, folds, k):
    accuracy_sum = 0
    x, y = shuffle(x, y, random_state=42)
    fold_size = len(x)//folds
    print("\nFold Size: " + str(fold_size))
    for i in range(folds):
        start_index = i * fold_size
        end_index = (i+1) * fold_size if i < (folds - 1) else len(x)
        x_val_fold = x[start_index:end_index]
        x_train_fold = np.concatenate((x[:start_index], x[end_index:]))
        y_val_fold = y[start_index:end_index]
        y_train_fold = np.concatenate((y[:start_index], y[end_index:]))
        knn_classifier = KNN(k)
        knn_classifier.fit(x_train_fold, y_train_fold)
        predictions = knn_classifier.predict(x_val_fold)
        confusion_matrix = cross_validation(predictions, y_val_fold)
        accuracy_score = (confusion_matrix[0] + confusion_matrix[1]) / (len(y_val_fold))
        accuracy_sum += accuracy_score
        print("k: {} ====== fold index: {} ==== accuracy: {}".format(k, i, accuracy_score))
    accuracy = accuracy_sum / folds
    print("Accuracy: " + str(accuracy))
    return accuracy


from sklearn.feature_selection import SelectKBest, chi2

k_best = 2000  # Number of features to select
selector = SelectKBest(chi2, k=k_best)
X_train = selector.fit_transform(X_train, Y_train)
folds = 6
k_fold_cross_validation(X_train, Y_train, folds, k=191)

Test_file = "Test_data.txt"
preprocessed_test_data = []

with open(Test_file, 'r', encoding='utf-8') as file:
  for line in file:
    if line.strip():
      review_text = preprocess_text(line.strip())
      preprocessed_test_data.append(review_text)

X_test = tfidf_vectorizer.transform(preprocessed_test_data).toarray()
X_test = selector.transform(X_test)
best_k_from_trained_model = 191

knn_classifier = KNN(k=best_k_from_trained_model)
knn_classifier.fit(X_train, np.array(Y_train))

predictions = knn_classifier.predict(X_test)

# Replace the label value 2 with -1, as this was done earlier in the training phase
predictions[predictions == 2] = -1
file_name = "predictions.txt"
# Save the predictions to a file with a timestamped name
with open(file_name, 'w+') as file:
    for prediction in predictions:
        file.write(str(prediction) + '\n')

print(f"Predictions written to file: {file_name}")