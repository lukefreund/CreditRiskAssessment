from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data_collection.data_source import load_data, data_preprocessing, split_data, scale_features
import joblib

data = load_data('../data/credit_risk_dataset.csv').pipe(data_preprocessing).pipe(scale_features)

X_train, X_test, y_train, y_test = split_data(data, 'loan_status')

# Initialize the model
clf = RandomForestClassifier()

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

joblib.dump(clf, '../flask_credit_app/models/credit_risk_model.pkl')
