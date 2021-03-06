import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-17.549685993431947
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=make_pipeline(
            StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.55, tol=0.001)),
            StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.85, learning_rate=1.0, loss="lad", max_depth=4, max_features=0.6000000000000001, min_samples_leaf=3, min_samples_split=20, n_estimators=100, subsample=0.1)),
            MinMaxScaler(),
            LassoLarsCV(normalize=True)
        )),
        FunctionTransformer(copy)
    ),
    LinearSVR(C=5.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
