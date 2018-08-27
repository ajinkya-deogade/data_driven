import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-17.779179815808213
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=1.0, tol=0.001)),
    LinearSVR(C=5.0, dual=True, epsilon=1.0, loss="epsilon_insensitive", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
