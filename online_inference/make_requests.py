import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    data = data.iloc[:, :-1]
    request_features = list(data.columns)
    for i in range(data.shape[0]):
        request_data = data.iloc[i].to_list()
        req = {"data": [request_data], "features": request_features}
        response = requests.post("http://127.0.0.1:8000/predict/", json=req)
        print(response.status_code)
        print(response.json())

    