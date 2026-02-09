from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------- SAMPLE DATA --------
texts = [
    "Win a free phone now",
    "Limited offer click here",
    "Meeting at 10 am",
    "Project discussion today",
    "I love this movie",
    "This product is bad",
    "Amazing experience",
    "Worst service"
]

labels = [
    "spam", "spam",
    "ham", "ham",
    "positive", "negative",
    "positive", "negative"
]

# -------- TF-IDF --------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# -------- MODEL --------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------- PREDICTION --------
y_pred = model.predict(X_test)

# -------- RESULTS --------
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------- CUSTOM TEST --------
test_text = ["Congratulations! You won a prize"]
test_vector = vectorizer.transform(test_text)
print("\nPrediction:", model.predict(test_vector)[0])
