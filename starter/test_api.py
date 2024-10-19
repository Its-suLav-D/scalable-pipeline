import requests
import json

# API endpoint
url = "https://census-income-prediction-api.onrender.com/predict"

# Sample data for prediction
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Making the POST request
try:
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("Prediction successful!")
        print(f"Income prediction: {result['prediction']}")
    else:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred while making the request: {e}")

# Optional: Print the full response for debugging
print("\nFull response:")
print(json.dumps(response.json(), indent=2))
