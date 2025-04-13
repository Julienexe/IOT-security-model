from fastapi import FastAPI, HTTPException
import numpy as np
import torch
from typing import List, Dict, Any

# Use forward slashes or os.path for platform independence
model = torch.jit.load("app/enhanced_student_ssl_dqn_model.pt")
model.eval()

class_names = np.array(['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb', 'ARP_poisioning',
       'DDOS_Slowloris', 'DOS_SYN_Hping', 'Metasploit_Brute_Force_SSH',
       'NMAP_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_scan',
       'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN'])

app = FastAPI(title="IoT Network Intrusion Detection API")

@app.get('/')
def read_root():
    return {'message': 'IoT Network Intrusion Detection API'}

@app.post('/predict')
def predict(data:Dict) -> Dict[str, Any]:
    """
    Predicts the intrusion class of a given set of features.

    Args:
       data (dict): A dictionary containing the features to predict.
        e.g.

       {"features": [1.0, 2.0, 3.0, 4.0]}

        Note: The features must be in the same order as the training data with 84 features and no missing values.
        Training data is specified in the readme file.

    Returns:
        dict: A dictionary containing the predicted class name and detailed model outputs.
    """
    try:
        if 'features' not in data:
            raise HTTPException(status_code=400, detail="Missing 'features' in request body")
            
        # Extract features from the input dictionary
        features = data['features']
        
        # Convert to a 2D tensor by adding a batch dimension
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Model returns a dictionary with multiple tensors
            output_dict = model(input_tensor)
            
        # Process classification results
        classification = output_dict['classification'][0]  # Remove batch dimension
        predicted_idx = torch.argmax(classification).item()
        
        # Get the probabilities using softmax
        probabilities = torch.nn.functional.softmax(classification, dim=0).tolist()
        
        # Create a user-friendly response
        response = {
            'predicted_class': class_names[predicted_idx],
            'confidence': probabilities[predicted_idx],
            'class_probabilities': dict(zip(class_names.tolist(), probabilities)),
        }
        
        # Add raw model outputs if needed
        if 'include_raw_output' in data and data['include_raw_output']:
            response['raw_output'] = {
                'classification': output_dict['classification'][0].tolist(),
                'q_values': output_dict['q_values'][0].tolist() if 'q_values' in output_dict else None,
                'reconstruction': output_dict['reconstruction'][0].tolist() if 'reconstruction' in output_dict else None,
                'feature_prediction': output_dict['feature_prediction'][0].tolist() if 'feature_prediction' in output_dict else None
            }
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")