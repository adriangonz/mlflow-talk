import pandas as pd
import numpy as np
from seldon_core.seldon_client import SeldonClient

sc = SeldonClient(
    gateway="istio", 
    namespace="default",
    gateway_endpoint="localhost:8083",
    deployment_name='wines-classifier')

df = pd.read_csv("./training/wine-quality.csv")

def _get_reward(y, y_pred):
    if y == y_pred:
        return 500    
    
    return 1 / np.square(y - y_pred)

def _test_row(row):
    input_features = row[:-1]
    feature_names = input_features.index.to_list()
    X = input_features.values.reshape(1, -1)
    y = row[-1].reshape(1, -1)
    
    r = sc.predict(
        data=X,
        names=feature_names)
    
    y_pred = r.response['data']['tensor']['values']
    reward = _get_reward(y, y_pred)
    sc.feedback(
        prediction_request=r.request,
        prediction_response=r.response,
        reward=reward)
    
    return reward[0]

while True:
    df.apply(_test_row, axis=1)
