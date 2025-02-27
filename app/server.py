from fastapi import FastAPI
import pickle
import numpy as np  

model = pickle.load(open('app/enhanced_student_ssl_dqn_model.pkl', 'rb'))

class_names = np.array(['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb', 'ARP_poisioning',
       'DDOS_Slowloris', 'DOS_SYN_Hping', 'Metasploit_Brute_Force_SSH',
       'NMAP_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_scan',
       'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN'])

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message': 'This is the homepage of the API'}

@app.post('/predict')
def predict(data: dict):
    """Predict the class of the network traffic
       
         This function receives a JSON object with the network traffic features
            and returns the predicted class.
        
        Args:
            data (dict): The network traffic features 
            e.g. {'features': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
        Returns:
            dict: The predicted class

    """
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'prediction': class_name}