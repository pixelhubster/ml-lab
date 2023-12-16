from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean, std
# define the dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=7
)

# define the pipeline
steps = list()
steps.append(("scaler", MinMaxScaler()))
steps.append(("model", LogisticRegression()))
pipeline = Pipeline(steps=steps)

# cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# cross validation score
scores = cross_val_score(pipeline,X, y, cv=cv,scoring="accuracy", n_jobs=-1)

# print the mean and std score
print("Accuracy %.3f (%.3f)" %(mean(scores)*100, std(scores)*100))