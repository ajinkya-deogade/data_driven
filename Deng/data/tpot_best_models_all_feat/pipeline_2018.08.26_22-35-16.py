import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-19.230224345765613
exported_pipeline = LinearSVR(C=25.0, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
