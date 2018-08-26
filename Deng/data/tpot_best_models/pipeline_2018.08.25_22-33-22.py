import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-17.569998079087192
exported_pipeline = make_pipeline(
    ZeroCount(),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.01, loss="lad", max_depth=5, max_features=0.8500000000000001, min_samples_leaf=13, min_samples_split=14, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
