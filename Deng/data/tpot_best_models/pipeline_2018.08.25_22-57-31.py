import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-16.688023353137517
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=1, min_child_weight=16, n_estimators=100, nthread=1, subsample=0.55)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=0.001, loss="ls", max_depth=2, max_features=0.7500000000000001, min_samples_leaf=12, min_samples_split=17, n_estimators=100, subsample=1.0)),
    Nystroem(gamma=0.25, kernel="laplacian", n_components=10),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="lad", max_depth=2, max_features=0.8500000000000001, min_samples_leaf=19, min_samples_split=7, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
