from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
# define the dataset
X,y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=7
)

# scale
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# define model
model = LogisticRegression()
# define model: repeated stratified k-fold model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the model using cross-validation
scores  = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
print("Score: %.3f (%.3f)" %(mean(scores)*100, std(scores)*100))
print(scores)