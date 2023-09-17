import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    filepath = '../../data/credit_risk_dataset.csv'
    data = pd.read_csv(filepath)
    return data


def scale_features(data):
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data


def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


data = load_data()

# Fill in missing values with mode value.
data['person_emp_length'].fillna(data['person_emp_length'].mode()[0], inplace=True)
data['loan_int_rate'].fillna(data['loan_int_rate'].mode()[0], inplace=True)

# replace loan grade with numbers
grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
data['loan_grade'] = data['loan_grade'].map(grade_mapping)

# Convert categorical into numerical values
data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'cb_person_default_on_file'],
                      drop_first=True)

# Scale features
data = scale_features(data)

# Split the data
X_train, X_test, y_train, y_test = split_data(data, 'loan_status')

print(data.head())
print(data.describe())
print(data.info())



