from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import uvicorn
import json
from typing import Dict, List, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="IOT Intrusion Detection Model")

# Setup Jinja2 templates
BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))

# Mount static files
app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR, 'static'))), name="static")



model = torch.jit.load("app/enhanced_student_ssl_dqn_model.pt")
model.eval()

# Store prediction history
predictions = []

class_names = np.array(['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb', 'ARP_poisioning',
       'DDOS_Slowloris', 'DOS_SYN_Hping', 'Metasploit_Brute_Force_SSH',
       'NMAP_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_scan',
       'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN'])

# Sample common attack patterns for suggestions
common_patterns = [
    {
        "name": "Normal Traffic",
        "features": "0.1,0.0,0.0,0.0",
        "description": "Regular network traffic pattern"
    },
    {
        "name": "DoS Attack",
        "features": "0.9,0.1,0.8,0.0",
        "description": "Denial of Service attack pattern"
    },
    {
        "name": "Port Scan",
        "features": "0.2,0.9,0.3,0.1",
        "description": "Port scanning activity"
    },
    {
        "name": "Data Exfiltration",
        "features": "0.5,0.2,0.1,0.9",
        "description": "Potential data theft pattern"
    },
    {
        "name": "Brute Force",
        "features": "0.8,0.0,0.0,0.7",
        "description": "Brute force attack attempt"
    }
]



# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

# Route to handle the search suggestions
@app.post("/add-item", response_class=HTMLResponse)
async def add_item(request: Request, features_simple: str = Form(...)):
    # Filter suggestions based on input
    search_results = []
    
    if features_simple:
        for pattern in common_patterns:
            if (features_simple.lower() in pattern["name"].lower() or 
                features_simple in pattern["features"]):
                search_results.append(pattern)
    
    # Return HTML for search results
    return templates.TemplateResponse(
        "search-results.html", 
        {"request": request, "search_results": search_results}
    )

# Route to get prediction history
@app.get("/history", response_class=HTMLResponse)
async def get_history(request: Request):
    return templates.TemplateResponse(
        "prediction-history.html", 
        {"request": request, "predictions": predictions}
    )

# Prediction endpoint
@app.post('/predict')
async def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts the intrusion class of a given set of features.
    """
    try:
        if 'features' not in data:
            raise HTTPException(status_code=400, detail="Missing 'features' in request body")
            
        # Extract features from the input dictionary
        features = data['features']
        
        # Ensure we have the right number of features
        if len(features) < 84:
            # Pad with zeros if fewer than 84 features
            features = features + [0.0] * (84 - len(features))
        elif len(features) > 84:
            # Truncate if more than 84 features
            features = features[:84]
        
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
            'class_probabilities': dict(zip(class_names, probabilities)),
        }
        
        # Add raw model outputs if requested
        if 'include_raw_output' in data and data['include_raw_output']:
            response['raw_output'] = {
                'classification': output_dict['classification'][0].tolist(),
                'q_values': output_dict['q_values'][0].tolist() if 'q_values' in output_dict else None,
                'reconstruction': output_dict['reconstruction'][0].tolist() if 'reconstruction' in output_dict else None,
                'feature_prediction': output_dict['feature_prediction'][0].tolist() if 'feature_prediction' in output_dict else None
            }
        
        # Add to predictions history
        prediction_text = f"{response['predicted_class']} (Confidence: {response['confidence']:.2%})"
        if prediction_text not in predictions:
            predictions.append(prediction_text)
        if len(predictions) > 10:  # Keep only the last 10 predictions
            predictions.pop(0)
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
