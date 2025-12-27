# -*- coding: utf-8 -*-
"""
Servidor Flask para Clasificación de Biomas Amazónicos
Universidad Nacional del Altiplano - Ciencia de Datos

Uso:
    pip install flask flask-cors torch torchvision pillow
    python servidor.py
    
Luego abrir index.html en el navegador
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import time
import os

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde HTML

# ============================================
# CONFIGURACIÓN DEL MODELO
# ============================================

CLASSES = ['bark', 'canopy', 'leaf', 'profile', 'sap']
CLASS_NAMES = {
    'bark': {'es': 'Corteza', 'emoji': '🌳', 'desc': 'Textura de corteza de árbol amazónico'},
    'canopy': {'es': 'Dosel', 'emoji': '🌲', 'desc': 'Vista del follaje desde abajo'},
    'leaf': {'es': 'Hoja', 'emoji': '🍃', 'desc': 'Hoja individual de planta amazónica'},
    'profile': {'es': 'Perfil', 'emoji': '🪵', 'desc': 'Vista lateral del tronco y ramas'},
    'sap': {'es': 'Savia', 'emoji': '💧', 'desc': 'Resina o secreción del árbol'}
}

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================
# CARGAR MODELO
# ============================================

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES))
    )
    
    model_path = 'modelo_biomas_amazonia.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Modelo cargado: {model_path}")
    else:
        print(f"⚠ No se encontró {model_path} - usando modelo sin entrenar")
    
    model.eval()
    return model

print("Cargando modelo...")
model = load_model()

# ============================================
# RUTAS API
# ============================================

@app.route('/api/clasificar', methods=['POST'])
def clasificar():
    """Endpoint para clasificar imagen"""
    try:
        start_time = time.time()
        
        # Obtener imagen (base64 o archivo)
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream)
        elif request.json and 'image' in request.json:
            # Base64
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            return jsonify({'error': 'No se recibió imagen'}), 400
        
        # Convertir a RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocesar
        input_tensor = transform(image).unsqueeze(0)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Construir respuesta
        predictions = []
        for i, class_name in enumerate(CLASSES):
            predictions.append({
                'class': class_name,
                'name_es': CLASS_NAMES[class_name]['es'],
                'emoji': CLASS_NAMES[class_name]['emoji'],
                'description': CLASS_NAMES[class_name]['desc'],
                'probability': float(probs[i]),
                'percentage': round(float(probs[i]) * 100, 2)
            })
        
        # Ordenar por probabilidad
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': predictions[0],
            'inference_time_ms': round(inference_time, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Verificar que el servidor está funcionando"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': CLASSES
    })

@app.route('/')
def home():
    return """
    <h1>🌿 API Clasificador de Biomas Amazónicos</h1>
    <p>Endpoints:</p>
    <ul>
        <li>POST /api/clasificar - Enviar imagen para clasificar</li>
        <li>GET /api/health - Verificar estado del servidor</li>
    </ul>
    <p>Abrir <a href="index.html">index.html</a> para usar la interfaz gráfica.</p>
    """

# ============================================
# EJECUTAR SERVIDOR
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌿 Servidor de Clasificación de Biomas")
    print("="*50)
    print("\n📡 API corriendo en: http://localhost:5000")
    print("📄 Abrir index.html en el navegador\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
