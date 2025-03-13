import requests
import cv2
import numpy as np

response = requests.post("http://0.0.0.0:9988/diffusion_real")
response = response.json()

# actions = response["actions"]
# actions = np.array(actions)
# print(actions)
print(response)
