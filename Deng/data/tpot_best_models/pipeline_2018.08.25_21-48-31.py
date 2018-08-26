import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-18.11898070399461
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.01, loss="huber", max_depth=10, max_features=0.15000000000000002, min_samples_leaf=5, min_samples_split=16, n_estimators=100, subsample=0.2)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)