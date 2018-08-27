import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-15.168807061620631
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="quantile", max_depth=1, max_features=0.35000000000000003, min_samples_leaf=10, min_samples_split=20, n_estimators=100, subsample=0.7500000000000001)),
    SelectFwe(score_func=f_regression, alpha=0.027),
    SelectPercentile(score_func=f_regression, percentile=72),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=9, n_estimators=100, nthread=1, subsample=0.15000000000000002)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
