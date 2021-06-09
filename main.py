import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read dataset
data = pd.read_csv('data/diabetes.csv')

# Separate attribute
x = data[data.columns[:8]]

# Separate label
y = data['Outcome']

# Standardize values from dataset
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Separate data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Create SVC object and train model
clf = SVC()
clf.fit(x_train, y_train)

# Show prediction accuracy score
print(clf.score(x_test, y_test))