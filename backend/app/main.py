from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from .model_utils import ModelManager  # <-- CORREGIDO: con punto
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hand Gesture Recognition API", version="1.0.0")

# Agregar ruta raíz
@app.get("/")
def read_root():
    return {
        "message": "Hand Gesture Recognition API", 
        "version": "1.0.0",
        "endpoints": {
            "upload_sample": "/api/upload_sample (POST)",
            "train": "/api/train (POST)", 
            "predict": "/api/predict (POST)",
            "list_models": "/api/models (GET)",
            "samples_info": "/api/samples (GET)",
            "delete_model": "/api/model/{model_name} (DELETE)",
            "docs": "/docs"
        }
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
manager = ModelManager()
COLLECTED = []  # in-memory list of (landmarks, label)

def extract_landmarks_from_image_bytes(image_bytes) -> list:
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(img)
        # convert RGB to BGR for cv2/mediapipe
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        with mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            results = hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                return []
            
            # Usar la primera mano detectada
            hand = results.multi_hand_landmarks[0]
            lm = []
            
            # Extraer coordenadas normalizadas
            for p in hand.landmark:
                lm.extend([p.x, p.y, p.z])
            
            # Añadir características adicionales para mejorar precisión
            if len(hand.landmark) >= 9:
                # Calcular distancia entre puntos clave
                thumb_tip = np.array([hand.landmark[4].x, hand.landmark[4].y, hand.landmark[4].z])
                index_tip = np.array([hand.landmark[8].x, hand.landmark[8].y, hand.landmark[8].z])
                distance = np.linalg.norm(thumb_tip - index_tip)
                lm.append(distance)
            
            return lm
    except Exception as e:
        logger.error(f"Error extracting landmarks: {str(e)}")
        return []

@app.post('/api/upload_sample')
async def upload_sample(label: str = Form(...), file: UploadFile = File(...)):
    try:
        content = await file.read()
        lm = extract_landmarks_from_image_bytes(content)
        if not lm:
            raise HTTPException(status_code=400, detail='No hand detected or low confidence')
        
        COLLECTED.append((lm, label))
        logger.info(f"Sample uploaded for label '{label}'. Total samples: {len(COLLECTED)}")
        
        return JSONResponse(
            status_code=200,
            content={
                'status': 'success', 
                'n_samples': len(COLLECTED),
                'message': f'Muestra para "{label}" agregada correctamente'
            }
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/train')
async def train_model(name: str = Form(...)):
    if not COLLECTED:
        raise HTTPException(status_code=400, detail='No samples collected')
    
    X = [c[0] for c in COLLECTED]
    y = [c[1] for c in COLLECTED]
    
    try:
        res = manager.train_and_save(X, y, name)
        logger.info(f"Model '{name}' trained with accuracy: {res['accuracy']}")
        
        return JSONResponse(
            status_code=200,
            content={
                'status': 'success', 
                'accuracy': res['accuracy'], 
                'n_samples': res['n_samples'],
                'report': res['classification_report'],
                'message': f'Modelo "{name}" entrenado con {res["accuracy"]*100:.2f}% de precisión'
            }
        )
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/models')
async def list_models():
    try:
        models = manager.list_models()
        models_info = []
        
        for model_name in models:
            info = manager.get_model_info(model_name)
            models_info.append({
                'name': model_name,
                'accuracy': info.get('accuracy', 0),
                'n_samples': info.get('n_samples', 0),
                'classes': list(info.get('n_samples_per_class', {}).keys())
            })
        
        return {'models': models_info}
    except Exception as e:
        logger.error(f"List models error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/model/{model_name}')
async def get_model_details(model_name: str):
    try:
        info = manager.get_model_info(model_name)
        if not info:
            raise HTTPException(status_code=404, detail='Model not found')
        
        return info
    except Exception as e:
        logger.error(f"Model details error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/model/{model_name}')
async def delete_model(model_name: str):
    """Eliminar un modelo específico y sus metadatos"""
    try:
        # Verificar si el modelo existe
        models = manager.list_models()
        if model_name not in models:
            raise HTTPException(status_code=404, detail='Model not found')
        
        # Obtener rutas de archivos
        model_path = os.path.join(manager.models_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(manager.metadata_dir, f"{model_name}_metadata.json")
        
        # Eliminar archivos
        files_deleted = []
        if os.path.exists(model_path):
            os.remove(model_path)
            files_deleted.append('modelo')
            logger.info(f"Model file deleted: {model_path}")
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            files_deleted.append('metadatos')
            logger.info(f"Metadata file deleted: {metadata_path}")
        
        if not files_deleted:
            raise HTTPException(status_code=404, detail='No files found to delete')
        
        return JSONResponse(
            status_code=200,
            content={
                'status': 'success',
                'message': f'Modelo "{model_name}" eliminado exitosamente',
                'files_deleted': files_deleted
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f'Error deleting model: {str(e)}')

@app.post('/api/predict')
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    try:
        content = await file.read()
        lm = extract_landmarks_from_image_bytes(content)
        if not lm:
            raise HTTPException(status_code=400, detail='No hand detected or low confidence')
        
        result = manager.predict_with_confidence(lm, model)
        
        return JSONResponse(
            status_code=200,
            content={
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result['probabilities'],
                'all_predictions': result['all_predictions'],
                'message': f'Predicción: {result["prediction"]} ({result["confidence"]*100:.2f}% de confianza)'
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/samples')
async def get_samples_info():
    """Obtener información sobre las muestras recolectadas"""
    labels = [c[1] for c in COLLECTED]
    unique_labels = set(labels)
    
    return {
        'total_samples': len(COLLECTED),
        'samples_per_class': {label: labels.count(label) for label in unique_labels}
    }

@app.delete('/api/clear_samples')
async def clear_samples():
    """Eliminar todas las muestras recolectadas"""
    global COLLECTED
    count = len(COLLECTED)
    COLLECTED = []
    logger.info(f"Cleared {count} samples")
    
    return {'status': 'success', 'message': f'{count} muestras eliminadas'}

# Para ejecutar directamente
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
