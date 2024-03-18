# gradient_boosting.py
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class GBClassification:
    def __init__(self, dataset_file):
        self.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        print("Using Gradient Boosting Algorithm for Classification")

        # Load the dataset from the CSV file
        dataset = pd.read_csv(dataset_file)

        # Split the dataset into features (X) and labels (y)
        X = dataset.drop(columns=["type"])
        y = dataset["type"]

        # Perform feature scaling on the input data using Min-Max scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the classifier
        self.clf.fit(X_scaled, y)

    def classify(self, data):
        # Perform feature scaling on the input data using Min-Max scaling
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Make predictions
        prediction = self.clf.predict(data_scaled)
        print("Classification Result:", prediction)

        if 1 in prediction:
            print("Abnormal traffic detected. Taking action to block...")
            # Replace this with your code to take action (e.g., block traffic)
            # You can add code here to perform actions such as blocking the traffic
            # For example, you can use Ryu to send OpenFlow messages to block traffic.

        return prediction
