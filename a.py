from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on training data
y_pred = clf.predict(X_train)

# Generate confusion matrix
cm = confusion_matrix(y_train, y_pred)
print(cm)