import numpy as np
# from sklearn import preprocessing
# from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
feature = np.array([
    [-500.5],
    [-100.1],
    [0],
    [100.1],
    [900.9]
])
# Scaling using Min Max Scaler: 
#           this is done using the formular y = (x-min)/(max -min)  
# it scales the values in the range of 0 and 1
# minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
# scaled_features = minmax_scale.fit_transform(feature)
# print(scaled_features)

# Transforming using standard Scaler: this is done using the formular y = (x-mean)/(std)
# it transforms data such that the mean is 0 and the std is 1
# scaler = preprocessing.StandardScaler()
# standardized = scaler.fit_transform(feature)
# print(standardized.var())

# Robust Scaler Method
#   scales feature using the median and quartile range.
# robustScaler = preprocessing.RobustScaler()
# robust_scaled = robustScaler.fit_transform(feature)
# print(robust_scaled)

# Normalizer
# normalizer = Normalizer(norm='l1')
# print(normalizer.transform(feature))

# polynomial interaction is used when we want to include notion 
# that exits a nonlinear relationship btwn the features and the target
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
print(polynomial_interaction.fit_transform(feature))