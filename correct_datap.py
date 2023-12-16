from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# define the dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=7
)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=1)

# scale our training set and transform the training & testing dataset
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_transform = scaler.transform(X_train)
X_test_transform = scaler.transform(X_test)

# define the model
model = LogisticRegression()
model.fit(X_train_transform, y_train)
yhat = model.predict(X_test_transform)

# accuracy check
acc = accuracy_score(y_test, yhat)
print("Accuracy %.3f" %(acc*100))