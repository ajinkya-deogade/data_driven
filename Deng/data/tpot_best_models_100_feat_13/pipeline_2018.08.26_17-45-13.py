import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-15.203873985130153
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.026000000000000002),
    StackingEstimator(estimator=LinearSVR(C=10.0, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.01)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.001, loss="quantile", max_depth=2, max_features=0.35000000000000003, min_samples_leaf=15, min_samples_split=17, n_estimators=100, subsample=0.7500000000000001)),
    Normalizer(norm="l1"),
    XGBRegressor(learning_rate=0.01, max_depth=6, min_child_weight=8, n_estimators=100, nthread=1, subsample=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
