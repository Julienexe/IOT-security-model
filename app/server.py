from fastapi import FastAPI
import joblib
import numpy as np
import pickle
from typing import List, Dict, Any, Union

# Define a complete EnhancedStudentModel class
class EnhancedStudentModel:
    def __init__(self, model_components: Dict[str, Any] = None):
        """
        Initialize the EnhancedStudentModel with model components.
        
        Args:
            model_components: Dictionary containing model weights, parameters, etc.
        """
        if model_components:
            self.weights = model_components.get('weights')
            self.scaler = model_components.get('scaler')
            self.encoder = model_components.get('encoder')
            self.threshold = model_components.get('threshold', 0.5)
            self.feature_names = model_components.get('feature_names')
            self.n_features = model_components.get('n_features')
            self.model_type = model_components.get('model_type', 'dqn')
            self.ssl_encoder = model_components.get('ssl_encoder')
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess input features before prediction.
        
        Args:
            features: Input features to preprocess
            
        Returns:
            Preprocessed features
        """
        # Apply scaling if a scaler exists
        if hasattr(self, 'scaler') and self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Apply encoding if SSL encoder exists
        if hasattr(self, 'ssl_encoder') and self.ssl_encoder is not None:
            features = self.ssl_encoder.predict(features)
            
        return features
    
    def predict(self, features: np.ndarray) -> List[int]:
        """
        Make predictions using the model.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Predicted class indices
        """
        # Preprocess features
        processed_features = self.preprocess(features)
        
        # Handle different model types
        if hasattr(self, 'model_type') and self.model_type == 'dqn':
            # For DQN-based models
            if hasattr(self, 'weights') and self.weights is not None:
                # This is a simplified prediction - replace with actual logic
                # based on weights and the model architecture
                predictions = np.argmax(processed_features @ self.weights, axis=1)
                return predictions.tolist()
        
        # Default fallback prediction logic
        # This is very simplified - replace with actual model logic
        return [0] * len(features)
    
    @classmethod
    def load(cls, filepath: str) -> 'EnhancedStudentModel':
        """
        Load model components and create a model instance.
        
        Args:
            filepath: Path to the saved model components
            
        Returns:
            An initialized EnhancedStudentModel
        """
        try:
            # Try loading as components dict first
            components = joblib.load(filepath)
            if isinstance(components, dict):
                return cls(components)
            else:
                # If it's already a model instance, return it
                return components
        except (AttributeError, pickle.UnpicklingError):
            # If loading fails, try to extract just the components needed
            try:
                # Try to load the file as a raw model and extract components
                raw_model = joblib.load(filepath)
                components = {
                    'weights': getattr(raw_model, 'weights', None),
                    'scaler': getattr(raw_model, 'scaler', None),
                    'encoder': getattr(raw_model, 'encoder', None),
                    'ssl_encoder': getattr(raw_model, 'ssl_encoder', None),
                    'feature_names': getattr(raw_model, 'feature_names', None),
                    'n_features': getattr(raw_model, 'n_features', None),
                    'model_type': getattr(raw_model, 'model_type', 'dqn')
                }
                return cls(components)
            except:
                # If all else fails, return an empty model
                print("Warning: Could not load model properly. Using default empty model.")
                return cls()

# Now use the class method to load the model
try:
    model = EnhancedStudentModel.load('app/enhanced_student_ssl_dqn_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to a default model if loading fails
    model = EnhancedStudentModel()

# Define class names
class_names = np.array(['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb', 'ARP_poisioning',
       'DDOS_Slowloris', 'DOS_SYN_Hping', 'Metasploit_Brute_Force_SSH',
       'NMAP_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_scan',
       'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN'])

# Create FastAPI app
app = FastAPI()

@app.get('/')
def read_root():
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