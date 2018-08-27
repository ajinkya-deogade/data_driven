import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-16.28615353721678
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.05, kernel="laplacian", n_components=10),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.30000000000000004, tol=1e-05)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=13, min_samples_split=11, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="lad", max_depth=1, max_features=0.6000000000000001, min_samples_leaf=13, min_samples_split=14, n_estimators=100, subsample=0.5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
