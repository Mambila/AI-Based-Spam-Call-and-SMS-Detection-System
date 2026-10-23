import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

def train_spam_model():
    print("Loading augmented dataset...")
    # Load the combined dataset (public data + local Zambian patterns)[cite: 1]
    df = pd.read_csv('data/processed/augmented_spam_dataset.csv')
    
    # Map labels to binary: ham = 0, spam = 1
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    # 2. Feature Extraction (TF-IDF)[cite: 1]
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['message'])
    y = df['label_num']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Address Class Imbalance with SMOTE[cite: 1]
    print("Applying SMOTE resampling...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 5. Train Logistic Regression Model[cite: 1]
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_resampled, y_train_resampled)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Ham (Safe)', 'Spam']))

    # 7. Save Model and Vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/spam_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("\nModel and vectorizer saved successfully in the 'models/' folder!")

if __name__ == "__main__":
    train_spam_model()