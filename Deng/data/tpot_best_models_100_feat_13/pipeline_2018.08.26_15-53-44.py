import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-15.343469241217278
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=0.01, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.01)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.001, loss="quantile", max_depth=2, max_features=0.35000000000000003, min_samples_leaf=14, min_samples_split=17, n_estimators=100, subsample=0.7500000000000001)),
    XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=9, n_estimators=100, nthread=1, subsample=0.15000000000000002)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
