import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-15.477534747042629
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    SelectFwe(score_func=f_regression, alpha=0.021),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.001, loss="ls", max_depth=6, max_features=0.6500000000000001, min_samples_leaf=10, min_samples_split=2, n_estimators=100, subsample=0.15000000000000002)),
    StackingEstimator(estimator=RidgeCV()),
    XGBRegressor(learning_rate=0.01, max_depth=2, min_child_weight=9, n_estimators=100, nthread=1, subsample=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
