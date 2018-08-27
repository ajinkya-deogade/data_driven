import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-17.4072236479251
exported_pipeline = make_pipeline(
    FeatureAgglomeration(affinity="l2", linkage="average"),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.55, min_samples_leaf=14, min_samples_split=8, n_estimators=100)),
    ZeroCount(),
    VarianceThreshold(threshold=0.0001),
    OneHotEncoder(minimum_fraction=0.05, sparse=False),
    LinearSVR(C=0.5, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
