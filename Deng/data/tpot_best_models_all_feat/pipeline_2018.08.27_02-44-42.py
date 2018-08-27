import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-17.339244629634862
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=25),
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.15000000000000002, n_estimators=100), threshold=0.05),
    StackingEstimator(estimator=LinearSVR(C=0.5, dual=True, epsilon=1.0, loss="squared_epsilon_insensitive", tol=0.001)),
    StackingEstimator(estimator=LinearSVR(C=0.001, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.001)),
    LinearSVR(C=25.0, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=0.0001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
