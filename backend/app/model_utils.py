import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from joblib import dump, load
import json
from datetime import datetime  # te faltaba importar esto

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
METADATA_DIR = os.path.join(os.path.dirname(__file__), "model_metadata")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(METADATA_DIR):
    os.makedirs(METADATA_DIR)

class ModelManager:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.metadata_dir = METADATA_DIR

    def _model_path(self, name: str):
        if not name:
            name = "model"
        name = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.models_dir, f"{name}.joblib")
    
    def _metadata_path(self, name: str):
        if not name:
            name = "model"
        name = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.metadata_dir, f"{name}_metadata.json")

    def save_model(self, model, name: str, metadata: dict = None):
        dump(model, self._model_path(name))
        if metadata:
            with open(self._metadata_path(name), 'w') as f:
                json.dump(metadata, f)

    def load_model(self, name: str):
        return load(self._model_path(name))
    
    def load_metadata(self, name: str):
        metadata_path = self._metadata_path(name)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def list_models(self):
        files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        return [os.path.splitext(f)[0] for f in files]

    def train_and_save(self, X, y, name: str, test_size=0.2, random_state=42):
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 10:
            raise ValueError("Se necesitan al menos 10 muestras para entrenar.")
        if len(set(y)) < 2:
            raise ValueError("Se necesitan al menos 2 clases diferentes para entrenar.")
        
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X, y = smote.fit_resample(X, y)
        except ImportError:
            print("SMOTE no disponible, continuando sin balanceo de clases")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        cv_scores = cross_val_score(clf, X, y, cv=5)
        
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metadata = {
            'accuracy': float(score),
            'cross_val_mean': float(np.mean(cv_scores)),
            'cross_val_std': float(np.std(cv_scores)),
            'n_samples': len(X),
            'n_samples_per_class': {label: int(np.sum(y == label)) for label in np.unique(y)},
            'classification_report': report,
            'feature_importance': clf.feature_importances_.tolist(),
            'classes': clf.classes_.tolist(),
            'oob_score': float(clf.oob_score_) if hasattr(clf, 'oob_score_') else None,
            'training_date': datetime.now().isoformat()
        }
        
        self.save_model(clf, name, metadata)
        return metadata

    def predict_with_confidence(self, X, name: str):
        clf = self.load_model(name)
        metadata = self.load_metadata(name)
        
        probabilities = clf.predict_proba([X])[0]
        
        class_probabilities = {
            clf.classes_[i]: float(probabilities[i]) 
            for i in range(len(clf.classes_))
        }
        
        sorted_probs = sorted(
            class_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        prediction = sorted_probs[0][0]
        confidence = sorted_probs[0][1]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': class_probabilities,
            'all_predictions': sorted_probs,
            'metadata': metadata
        }
    
    def get_model_info(self, name: str):
        """Obtener información detallada de un modelo"""
        metadata = self.load_metadata(name)
        return metadata
