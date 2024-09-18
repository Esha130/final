import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load data
@st.cache
def load_data():
    data = pd.read_csv('reviews.csv')
    data_clean = data.dropna().drop_duplicates()
    return data_clean

# Function to train model
def train_models(data_clean):
    X = data_clean['Review']  # Features (reviews)
    y = data_clean['Overall_Rating']  # Target (rating)

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to numerical features using Tfidf
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_vec.toarray())
    X_test_scaled = scaler.transform(X_test_vec.toarray())

    # Train logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)

    # Train decision tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_scaled, y_train)
    y_pred_tree = decision_tree.predict(X_test_scaled)

    return logreg, decision_tree, vectorizer, scaler

# Visualize top cities
def plot_top_cities(data_clean):
    top_cities = data_clean['City'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    top_cities.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Cities with Most Restaurants')
    plt.ylabel('Number of Restaurants')
    plt.xlabel('City')
    st.pyplot(plt)

# Streamlit app layout
st.title('Restaurant Rating Prediction and City Insights')

# Load data
data_clean = load_data()

# Show dataset
if st.checkbox("Show raw data"):
    st.write(data_clean.head())

# Train models
logreg, decision_tree, vectorizer, scaler = train_models(data_clean)

# Make predictions
review_input = st.text_area("Enter a review to predict the rating:")

if review_input:
    review_vec = vectorizer.transform([review_input])
    review_scaled = scaler.transform(review_vec.toarray())

    # Logistic Regression Prediction
    logreg_pred = logreg.predict(review_scaled)
    st.write(f"Predicted Rating (Logistic Regression): {logreg_pred[0]}")

    # Decision Tree Prediction
    decision_tree_pred = decision_tree.predict(review_scaled)
    st.write(f"Predicted Rating (Decision Tree): {decision_tree_pred[0]}")

# Visualize top cities
if st.checkbox("Show Top Cities with Restaurants"):
    plot_top_cities(data_clean)
