import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import OneHotEncoder

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-16.842811812237162
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.2, sparse=False),
    Nystroem(gamma=0.05, kernel="laplacian", n_components=8),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="lad", max_depth=5, max_features=0.35000000000000003, min_samples_leaf=13, min_samples_split=14, n_estimators=100, subsample=0.25)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
