import gradio as gr
import os
import tempfile
import shutil
import json
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import threading
import time
import locale
import pwd
import grp
import stat
from pathlib import Path
import keras

def set_user_permissions(path):
    """Set proper user permissions for files and directories"""
    try:
        # Get current user info
        current_user = pwd.getpwuid(os.getuid())
        uid = current_user.pw_uid
        gid = current_user.pw_gid

        # Set ownership and permissions recursively
        for root, dirs, files in os.walk(path):
            # Set directory permissions
            os.chown(root, uid, gid)
            os.chmod(root, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)  # 755

            # Set file permissions
            for file in files:
                file_path = os.path.join(root, file)
                os.chown(file_path, uid, gid)
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)  # 644
    except Exception as e:
        print(f"Warning: Could not set permissions for {path}: {e}")

# Language detection and translation system
def detect_language():
    """Detect user language from system locale, fallback to Spanish"""
    try:
        system_locale = locale.getdefaultlocale()[0]
        if system_locale and system_locale.startswith('en'):
            return 'en'
        else:
            return 'es'
    except:
        return 'es'

# Translation dictionaries
TRANSLATIONS = {
    'es': {
        'title': '🔬 Entrenador de Clasificadores con TensorFlow',
        'subtitle_professional': '🚀 Plataforma Profesional de Machine Learning',
        'subtitle_advanced': 'Entrenamiento avanzado de modelos de clasificación con arquitecturas EfficientNet, MobileNet y más',
        
        # Dataset validation
        'dataset_tab': 'Validación de Dataset',
        'upload_validate_title': 'Subir y Validar Dataset',
        'upload_validate_desc': 'Sube un archivo ZIP con carpetas organizadas por clase para validar la estructura',
        'upload_zip': 'Subir Dataset (ZIP)',
        'validate_button': 'Validar Dataset',
        'validation_result': 'Resultado de Validación',
        
        # Data Augmentation
        'augmentation_tab': 'Data Augmentation',
        'augmentation_title': 'Configurar Data Augmentation',
        'augmentation_desc': 'Configura las transformaciones que se aplicarán a las imágenes durante el entrenamiento',
        'use_augmentation': 'Usar Data Augmentation',
        'rotation_label': 'Rotación',
        'max_degrees': 'Grados máximos',
        'translation_label': 'Traslación',
        'translation_x_factor': 'Factor de traslación X',
        'translation_y_factor': 'Factor de traslación Y',
        'flip_label': 'Volteo',
        'horizontal_flip': 'Volteo horizontal',
        'vertical_flip': 'Volteo vertical',
        'zoom_label': 'Zoom',
        'zoom_range': 'Rango de zoom',
        'brightness_label': 'Brillo',
        'brightness_factor': 'Factor de brillo',
        
        # Training
        'training_tab': 'Entrenamiento',
        'training_title': 'Configurar y Entrenar Modelo',
        'training_desc': 'Selecciona el modelo base y configura los parámetros de entrenamiento',
        'model_selection': 'Selección de Modelo',
        'epochs_label': 'Épocas',
        'batch_size_label': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'start_training': 'Iniciar Entrenamiento',
        'training_output': 'Salida del Entrenamiento',
        
        # Metrics
        'metrics_tab': 'Visualización de Métricas',
        'metrics_title': 'Métricas de Entrenamiento',
        'metrics_desc': 'Visualiza las métricas de entrenamiento y validación',
        'load_metrics': 'Cargar Métricas',
        'metrics_plot': 'Gráficas de Métricas',
        
        # Quantization
        'quantization_tab': 'Cuantización TFLite',
        'quantization_title': 'Cuantizar Modelo a TFLite',
        'quantization_desc': 'Convierte el modelo entrenado a formato TFLite optimizado',
        'quantize_model': 'Cuantizar Modelo',
        'quantization_type': 'Tipo de Cuantización',
        'quantization_output': 'Resultado de Cuantización',
        
        # Inference
        'inference_tab': 'Inferencia',
        'inference_title': 'Realizar Inferencia',
        'inference_desc': 'Prueba el modelo entrenado con nuevas imágenes',
        'upload_image': 'Subir Imagen',
        'predict_button': 'Predecir',
        'prediction_result': 'Resultado de Predicción',
        
        # General
        'refresh_button': 'Actualizar',
        'download_model': 'Descargar Modelo',
        'gpu_status': '🔥 Estado GPU:',
        'gpu_enabled': 'Habilitada',
        'cpu_mode': 'Modo CPU',
        'version_info': 'v2.0 - TensorFlow Edition',
        'copyright_text': '© 2024 Rachael Vision - Tecnología de IA Avanzada'
    },
    'en': {
        'title': '🔬 TensorFlow Classification Trainer',
        'subtitle_professional': '🚀 Professional Machine Learning Platform',
        'subtitle_advanced': 'Advanced training of classification models with EfficientNet, MobileNet and more architectures',
        
        # Dataset validation
        'dataset_tab': 'Dataset Validation',
        'upload_validate_title': 'Upload and Validate Dataset',
        'upload_validate_desc': 'Upload a ZIP file with folders organized by class to validate structure',
        'upload_zip': 'Upload Dataset (ZIP)',
        'validate_button': 'Validate Dataset',
        'validation_result': 'Validation Result',
        
        # Data Augmentation
        'augmentation_tab': 'Data Augmentation',
        'augmentation_title': 'Configure Data Augmentation',
        'augmentation_desc': 'Configure transformations to be applied to images during training',
        'use_augmentation': 'Use Data Augmentation',
        'rotation_label': 'Rotation',
        'max_degrees': 'Max degrees',
        'translation_label': 'Translation',
        'translation_x_factor': 'Translation X factor',
        'translation_y_factor': 'Translation Y factor',
        'flip_label': 'Flip',
        'horizontal_flip': 'Horizontal flip',
        'vertical_flip': 'Vertical flip',
        'zoom_label': 'Zoom',
        'zoom_range': 'Zoom range',
        'brightness_label': 'Brightness',
        'brightness_factor': 'Brightness factor',
        
        # Training
        'training_tab': 'Training',
        'training_title': 'Configure and Train Model',
        'training_desc': 'Select base model and configure training parameters',
        'model_selection': 'Model Selection',
        'epochs_label': 'Epochs',
        'batch_size_label': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'start_training': 'Start Training',
        'training_output': 'Training Output',
        
        # Metrics
        'metrics_tab': 'Metrics Visualization',
        'metrics_title': 'Training Metrics',
        'metrics_desc': 'Visualize training and validation metrics',
        'load_metrics': 'Load Metrics',
        'metrics_plot': 'Metrics Plots',
        
        # Quantization
        'quantization_tab': 'TFLite Quantization',
        'quantization_title': 'Quantize Model to TFLite',
        'quantization_desc': 'Convert trained model to optimized TFLite format',
        'quantize_model': 'Quantize Model',
        'quantization_type': 'Quantization Type',
        'quantization_output': 'Quantization Output',
        
        # Inference
        'inference_tab': 'Inference',
        'inference_title': 'Perform Inference',
        'inference_desc': 'Test trained model with new images',
        'upload_image': 'Upload Image',
        'predict_button': 'Predict',
        'prediction_result': 'Prediction Result',
        
        # General
        'refresh_button': 'Refresh',
        'download_model': 'Download Model',
        'gpu_status': '🔥 GPU Status:',
        'gpu_enabled': 'Enabled',
        'cpu_mode': 'CPU Mode',
        'version_info': 'v2.0 - TensorFlow Edition',
        'copyright_text': '© 2024 Rachael Vision - Advanced AI Technology'
    }
}

# Get current language and translation function
current_lang = detect_language()
def t(key):
    return TRANSLATIONS[current_lang].get(key, key)

# Global variables for model and training state
current_model = None
current_dataset_path = None
training_history = None
class_names = []
current_training_params = {}

def ResNet18(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):
    """Custom ResNet18 implementation for TensorFlow/Keras"""
    
    def identity_block(input_tensor, kernel_size, filters, stage, block):
        filters1, filters2 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        x = tf.keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
        
        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
        
        shortcut = tf.keras.layers.Conv2D(filters2, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1')(shortcut)
        
        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=(224, 224, 3))
    else:
        img_input = input_tensor
    
    # Initial layers
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # ResNet18 blocks
    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')
    
    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')
    
    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')
    
    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')
    
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    
    model = tf.keras.models.Model(img_input, x, name='resnet18')
    
    # Note: We can't load ImageNet weights for custom ResNet18 easily
    # You would need to implement weight loading from a pre-trained model
    
    return model

# Supported models configuration
MODEL_CONFIGS = {
    'ResNet18': {'size': 224, 'module': ResNet18, 'quantizable': False},
    'ResNet50': {'size': 224, 'module': tf.keras.applications.ResNet50, 'quantizable': True},
    'ResNet101': {'size': 224, 'module': tf.keras.applications.ResNet101, 'quantizable': True},
    'ResNet152': {'size': 224, 'module': tf.keras.applications.ResNet152, 'quantizable': True},
    'EfficientNetB0': {'size': 224, 'module': tf.keras.applications.EfficientNetB0, 'quantizable': True},
    'EfficientNetB1': {'size': 240, 'module': tf.keras.applications.EfficientNetB1, 'quantizable': True},
    'EfficientNetB2': {'size': 260, 'module': tf.keras.applications.EfficientNetB2, 'quantizable': True},
    'EfficientNetB3': {'size': 300, 'module': tf.keras.applications.EfficientNetB3, 'quantizable': True},
    'EfficientNetV2B0': {'size': 224, 'module': tf.keras.applications.EfficientNetV2B0, 'quantizable': True},
    'EfficientNetV2B1': {'size': 240, 'module': tf.keras.applications.EfficientNetV2B1, 'quantizable': True},
    'EfficientNetV2B2': {'size': 260, 'module': tf.keras.applications.EfficientNetV2B2, 'quantizable': True},
    'EfficientNetV2B3': {'size': 300, 'module': tf.keras.applications.EfficientNetV2B3, 'quantizable': True},
    'MobileNet': {'size': 224, 'module': tf.keras.applications.MobileNet, 'quantizable': True},
    'MobileNetV2': {'size': 224, 'module': tf.keras.applications.MobileNetV2, 'quantizable': True},
}

def check_model_availability():
    """Check which models are actually available in current TensorFlow version"""
    available_models = {}
    for model_name, config in MODEL_CONFIGS.items():
        try:
            # Try to access the model class
            model_class = config['module']
            if hasattr(model_class, '__call__'):  # Check if it's callable
                available_models[model_name] = config
        except (AttributeError, ImportError) as e:
            print(f"⚠️ Warning: Model {model_name} not available in this TensorFlow version: {e}")
            continue
    return available_models

def get_model_choices():
    """Generate model dropdown choices"""
    available_models = check_model_availability()
    return list(available_models.keys())

def get_model_name_from_choice(choice):
    """Extract actual model name from dropdown choice"""
    return choice

def validate_dataset(zip_file):
    """Validate uploaded dataset structure"""
    if zip_file is None:
        return "❌ No se ha subido ningún archivo ZIP"
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the root directory with classes
        root_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        
        if len(root_dirs) == 1:
            # Single root directory
            dataset_root = os.path.join(temp_dir, root_dirs[0])
        else:
            # Multiple directories at root level
            dataset_root = temp_dir
        
        # Get class directories
        class_dirs = [d for d in os.listdir(dataset_root) 
                     if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')]
        
        if len(class_dirs) < 2:
            shutil.rmtree(temp_dir)
            return "❌ Se necesitan al menos 2 clases para entrenamiento"
        
        # Count images per class
        results = []
        total_images = 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for class_name in sorted(class_dirs):
            class_path = os.path.join(dataset_root, class_name)
            images = [f for f in os.listdir(class_path) 
                     if os.path.splitext(f.lower())[1] in valid_extensions]
            
            image_count = len(images)
            total_images += image_count
            
            status = "✅" if image_count >= 10 else "⚠️"
            results.append(f"{status} {class_name}: {image_count} imágenes")
        
        # Store dataset path globally
        global current_dataset_path, class_names
        current_dataset_path = dataset_root
        class_names = sorted(class_dirs)
        
        summary = f"""✅ **Dataset Válido**

📊 **Resumen:**
- **Clases encontradas:** {len(class_dirs)}
- **Total de imágenes:** {total_images}
- **Promedio por clase:** {total_images // len(class_dirs)}

📁 **Detalles por clase:**
{chr(10).join(results)}

🎯 **Recomendaciones:**
- Mínimo recomendado: 50+ imágenes por clase
- Para mejores resultados: 100+ imágenes por clase
- Asegúrate de que las imágenes sean representativas

✅ **Dataset listo para entrenamiento!**"""
        
        return summary
        
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return f"❌ Error al procesar el dataset: {str(e)}"

def create_augmentation_pipeline(rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                                flip_h, flip_v, zoom_enabled, zoom_in_factor, zoom_out_factor,
                                brightness_enabled, brighten_factor, darken_factor):
    """Create data augmentation pipeline based on user settings"""
    layers = []
    
    if rotation_enabled and rotation_degrees > 0:
        layers.append(tf.keras.layers.RandomRotation(
            factor=rotation_degrees/180.0, fill_mode='nearest'
        ))
    
    if translation_enabled and (translation_x_factor > 0 or translation_y_factor > 0):
        layers.append(tf.keras.layers.RandomTranslation(
            height_factor=translation_y_factor, width_factor=translation_x_factor, fill_mode='nearest'
        ))
    
    if flip_h or flip_v:
        mode = 'horizontal_and_vertical' if (flip_h and flip_v) else ('horizontal' if flip_h else 'vertical')
        layers.append(tf.keras.layers.RandomFlip(mode=mode))
    
    if zoom_enabled and (zoom_in_factor > 0 or zoom_out_factor > 0):
        # Convert zoom factors to TensorFlow format
        max_zoom_in = zoom_in_factor  # zoom in means smaller values (crop more)
        max_zoom_out = zoom_out_factor  # zoom out means larger values (crop less)
        
        # Combine both zoom in and zoom out
        zoom_range = max(max_zoom_in, max_zoom_out)
        if zoom_range > 0:
            layers.append(tf.keras.layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range), 
                width_factor=(-zoom_range, zoom_range),
                fill_mode='nearest'
            ))
    
    # Custom brightness adjustment using RandomBrightness if available
    if brightness_enabled and (brighten_factor > 0 or darken_factor > 0):
        try:
            # Calculate brightness factor range
            max_brighten = brighten_factor
            max_darken = darken_factor
            brightness_range = max(max_brighten, max_darken)
            
            if brightness_range > 0:
                layers.append(tf.keras.layers.RandomBrightness(
                    factor=brightness_range, value_range=(0, 255)
                ))
        except AttributeError:
            # RandomBrightness not available in this TF version
            pass
    
    if layers:
        return tf.keras.Sequential(layers, name="img_augmentation")
    else:
        return tf.keras.Sequential([], name="img_augmentation")

def preview_augmentation_config(rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                               flip_h, flip_v, zoom_enabled, zoom_in_factor, zoom_out_factor,
                               brightness_enabled, brighten_factor, darken_factor):
    """Generate a preview of the augmentation configuration"""
    config_lines = ["📊 **Configuración de Data Augmentation**\n"]

    if rotation_enabled and rotation_degrees > 0:
        config_lines.append(f"🔄 **Rotación**: ±{rotation_degrees}° (activada)")
    else:
        config_lines.append("🔄 **Rotación**: Desactivada")

    if translation_enabled and (translation_x_factor > 0 or translation_y_factor > 0):
        config_lines.append(f"↔️ **Traslación X**: ±{translation_x_factor*100:.0f}% | **Y**: ±{translation_y_factor*100:.0f}% (activada)")
    else:
        config_lines.append("↔️ **Traslación**: Desactivada")

    flip_modes = []
    if flip_h:
        flip_modes.append("horizontal")
    if flip_v:
        flip_modes.append("vertical")

    if flip_modes:
        config_lines.append(f"🔄 **Volteos**: {', '.join(flip_modes)} (activados)")
    else:
        config_lines.append("🔄 **Volteos**: Desactivados")

    if zoom_enabled and (zoom_in_factor > 0 or zoom_out_factor > 0):
        zoom_details = []
        if zoom_in_factor > 0:
            zoom_details.append(f"acercamiento {zoom_in_factor*100:.0f}%")
        if zoom_out_factor > 0:
            zoom_details.append(f"alejamiento {zoom_out_factor*100:.0f}%")
        config_lines.append(f"🔍 **Zoom**: {', '.join(zoom_details)} (activado)")
    else:
        config_lines.append("🔍 **Zoom**: Desactivado")

    if brightness_enabled and (brighten_factor > 0 or darken_factor > 0):
        brightness_details = []
        if brighten_factor > 0:
            brightness_details.append(f"brighten {brighten_factor*100:.0f}%")
        if darken_factor > 0:
            brightness_details.append(f"darken {darken_factor*100:.0f}%")
        config_lines.append(f"☀️ **Brightness**: {', '.join(brightness_details)} (activado)")
    else:
        config_lines.append("☀️ **Brightness**: Desactivado")

    # Count active augmentations
    active_augmentations = sum([
        rotation_enabled and rotation_degrees > 0,
        translation_enabled and (translation_x_factor > 0 or translation_y_factor > 0),
        flip_h or flip_v,
        zoom_enabled and (zoom_in_factor > 0 or zoom_out_factor > 0),
        brightness_enabled and (brighten_factor > 0 or darken_factor > 0)
    ])

    config_lines.append(f"\n📈 **Total**: {active_augmentations} técnicas de augmentación activas")
    config_lines.append("✅ **Estado**: Configuración lista para entrenamiento")

    return "\n".join(config_lines)

def generate_augmentation_preview_grid(rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                                       flip_h, flip_v, zoom_enabled, zoom_in_factor, zoom_out_factor,
                                       brightness_enabled, brighten_factor, darken_factor):
    """Generate a 3x3 grid showing augmentation examples from dataset"""
    global current_dataset_path

    if current_dataset_path is None:
        return None

    try:
        # Find a random image from the dataset
        import random

        # Get all subdirectories (classes)
        class_dirs = [d for d in os.listdir(current_dataset_path)
                     if os.path.isdir(os.path.join(current_dataset_path, d))]

        if not class_dirs:
            return None

        # Pick a random class
        random_class = random.choice(class_dirs)
        class_path = os.path.join(current_dataset_path, random_class)

        # Get all images from that class
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            return None

        # Pick a random image
        random_image = random.choice(image_files)
        image_path = os.path.join(class_path, random_image)

        # Load image
        original_img = Image.open(image_path).convert('RGB')

        # Resize to standard size for display
        display_size = 224
        original_img = original_img.resize((display_size, display_size), Image.Resampling.LANCZOS)

        # Convert to numpy array for TensorFlow operations
        img_array = np.array(original_img).astype(np.float32)

        # Create augmentation pipeline
        augmentation = create_augmentation_pipeline(
            rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
            flip_h, flip_v, zoom_enabled, zoom_in_factor, zoom_out_factor,
            brightness_enabled, brighten_factor, darken_factor
        )

        # Generate 9 augmented versions (3x3 grid)
        augmented_images = []

        for i in range(9):
            # Apply augmentation
            aug_img = augmentation(img_array, training=True)
            aug_img = tf.clip_by_value(aug_img, 0, 255)
            aug_img = aug_img.numpy().astype(np.uint8)
            augmented_images.append(Image.fromarray(aug_img))

        # Create 3x3 grid
        grid_size = 3
        img_size = display_size
        grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size), color='white')

        for idx, img in enumerate(augmented_images):
            row = idx // grid_size
            col = idx % grid_size
            grid_img.paste(img, (col * img_size, row * img_size))

        return grid_img

    except Exception as e:
        print(f"Error generating augmentation preview: {e}")
        return None

def build_model(model_name, num_classes, img_size, augmentation_pipeline):
    """Build model with selected architecture"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo no soportado: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    base_model_class = config['module']
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    
    # Apply augmentation
    x = augmentation_pipeline(inputs)
    
    # Base model - handle ResNet18 without ImageNet weights
    if model_name == 'ResNet18':
        base_model = base_model_class(
            include_top=False,
            input_tensor=x,
            weights=None  # No pretrained weights available for custom ResNet18
        )
    else:
        base_model = base_model_class(
            include_top=False,
            input_tensor=x,
            weights='imagenet'
        )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model

def train_model(model_name_display, advanced_config, epochs, batch_size, frozen_lr, unfrozen_lr,
                rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                flip_h, flip_v, zoom_enabled, zoom_in_factor, zoom_out_factor,
                brightness_enabled, brighten_factor, darken_factor,
                progress=gr.Progress()):
    """Train the model with specified parameters"""
    global current_model, current_dataset_path, training_history, class_names

    # Training time tracking
    import time
    training_start_time = time.time()
    phase_start_time = time.time()

    # Custom callback for progress updates
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress_fn, total_epochs, phase_name, phase_start_progress, phase_end_progress):
            super().__init__()
            self.progress_fn = progress_fn
            self.total_epochs = total_epochs
            self.phase_name = phase_name
            self.phase_start_progress = phase_start_progress
            self.phase_end_progress = phase_end_progress
            self.epoch_start_time = None
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            # Calculate epoch time
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

            # Calculate progress within this phase
            epoch_progress = (epoch + 1) / self.total_epochs
            overall_progress = self.phase_start_progress + (epoch_progress * (self.phase_end_progress - self.phase_start_progress))

            # Calculate estimated time remaining
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.total_epochs - (epoch + 1)
            estimated_time = avg_epoch_time * remaining_epochs

            # Format time
            if estimated_time > 3600:
                time_str = f"{estimated_time/3600:.1f}h"
            elif estimated_time > 60:
                time_str = f"{estimated_time/60:.1f}min"
            else:
                time_str = f"{estimated_time:.0f}s"

            # Get metrics
            acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)

            # Update progress with detailed info
            desc = f"🔥 {self.phase_name} - Época {epoch + 1}/{self.total_epochs} | " \
                   f"Acc: {acc:.3f} | Val Acc: {val_acc:.3f} | " \
                   f"Loss: {loss:.3f} | ⏱️ Resta: ~{time_str}"

            self.progress_fn(overall_progress, desc=desc)
    
    if current_dataset_path is None:
        return "❌ Primero debes validar un dataset", None
    
    # Extract actual model name from display name
    model_name = get_model_name_from_choice(model_name_display)
    
    # Parameter logic based on advanced configuration
    if advanced_config:
        # Use manual configuration from UI
        epochs_total = int(epochs)
        epochs_frozen = (2 * epochs_total) // 3
        epochs_unfrozen = epochs_total - epochs_frozen
        frozen_learning_rate = frozen_lr
        finetune_learning_rate = unfrozen_lr
        manual_batch_size = batch_size
    else:
        # Use optimized automatic configuration
        if os.getenv('EPOCHS'):
            EPOCHS = int(os.getenv('EPOCHS'))
            if EPOCHS < 20 or EPOCHS > 100:
                EPOCHS = 40
        else:
            EPOCHS = 40
        
        epochs_total = int(EPOCHS)
        epochs_frozen = (2 * EPOCHS) // 3
        epochs_unfrozen = EPOCHS - epochs_frozen
        frozen_learning_rate = 0.01
        finetune_learning_rate = 0.0001
        manual_batch_size = None  # Will use dynamic calculation
    
    # Capture data augmentation configuration
    augmentation_config = {
        "rotation": {"enabled": rotation_enabled, "degrees": rotation_degrees},
        "translation": {"enabled": translation_enabled, "x_factor": translation_x_factor, "y_factor": translation_y_factor},
        "flip_horizontal": flip_h,
        "flip_vertical": flip_v,
        "zoom": {"enabled": zoom_enabled, "in_factor": zoom_in_factor, "out_factor": zoom_out_factor},
        "brightness": {"enabled": brightness_enabled, "brighten_factor": brighten_factor, "darken_factor": darken_factor}
    }
    
    last_progress_epoch = 0
    # Set final epochs value
    epochs = epochs_total
    
    try:
        # Set umask to ensure proper file permissions
        old_umask = os.umask(0o022)  # This will create files with 644 permissions

        progress(0.1, desc="Preparando dataset...")

        # Get model configuration
        config = MODEL_CONFIGS[model_name]
        img_size = config['size']
        IMG_SIZE = img_size  # For compatibility with user's logic
        
        # Batch size logic based on configuration mode
        if advanced_config and manual_batch_size is not None:
            # Use manual batch size from UI, but still apply safety limits for large images
            batch_size = manual_batch_size
            if IMG_SIZE > 456 and batch_size > 16:
                batch_size = 16  # Safety limit for very large images
            elif IMG_SIZE > 350 and batch_size > 32:
                batch_size = 32  # Safety limit for large images
        else:
            # Use automatic dynamic batch size logic
            batch_size = 64
            if IMG_SIZE > 350:
                batch_size = 32
                if IMG_SIZE > 456:
                    batch_size = 16
        
        # Create datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            current_dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="categorical"
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            current_dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="categorical"
        )
        
        # Performance optimization
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        progress(0.2, desc="Construyendo modelo...")
        
        # Create augmentation pipeline
        augmentation = create_augmentation_pipeline(
            rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
            flip_h, flip_v, zoom_enabled, zoom_in_factor, zoom_out_factor,
            brightness_enabled, brighten_factor, darken_factor
        )

        # Create strategy and build model within its scope (cloud-style)
        strategy = tf.distribute.MirroredStrategy()  # aunque tengas 1 GPU
        with strategy.scope():
            # Build model
            num_classes = len(class_names)
            model, base_model = build_model(model_name, num_classes, img_size, augmentation)

            # Compile model with configured learning rate for frozen phase
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=frozen_learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        progress(0.3, desc="Iniciando entrenamiento (fase frozen)...")

        # Phase 1: Train with frozen base model
        frozen_epochs = epochs_frozen

        # Create progress callback for frozen phase
        frozen_callback = ProgressCallback(
            progress_fn=progress,
            total_epochs=frozen_epochs,
            phase_name="Fase Frozen (Base Congelada)",
            phase_start_progress=0.3,
            phase_end_progress=0.6
        )

        history1 = model.fit(
            train_ds,
            epochs=frozen_epochs,
            validation_data=val_ds,
            verbose=0,  # Disable default verbose output
            callbacks=[frozen_callback]
        )
        
        # Update progress tracking
        last_progress_epoch = frozen_epochs
        
        progress(0.6, desc="Iniciando entrenamiento (fase unfrozen)...")

        # Fase 2: Fine-tune "estilo nube" (últimas 20 del modelo, BN congelada)
        for l in model.layers:        # congela todo
            l.trainable = False
        for l in model.layers[-20:]:  # últimas 20 entrenables salvo BN
            if not isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        unfrozen_epochs = epochs_unfrozen
        if unfrozen_epochs > 0:
            # Create progress callback for unfrozen phase
            unfrozen_callback = ProgressCallback(
                progress_fn=progress,
                total_epochs=unfrozen_epochs,
                phase_name="Fase Fine-Tuning (Ajuste Fino)",
                phase_start_progress=0.6,
                phase_end_progress=0.9
            )

            history2 = model.fit(
                train_ds,
                epochs=unfrozen_epochs,
                validation_data=val_ds,
                initial_epoch=0,  # Start unfrozen phase from epoch 0
                verbose=0,  # Disable default verbose output
                callbacks=[unfrozen_callback]
            )
            
            # Update progress tracking
            last_progress_epoch = frozen_epochs + unfrozen_epochs
            
            # Combine histories
            combined_history = {}
            for key in history1.history.keys():
                if key in history2.history:
                    combined_history[key] = history1.history[key] + history2.history[key]
                else:
                    combined_history[key] = history1.history[key]
            
            # Store individual phases for reference script compatibility
            frozen_history = history1.history
            unfrozen_history = history2.history
        else:
            combined_history = history1.history
            frozen_history = history1.history
            unfrozen_history = None  # No unfrozen phase
        
        progress(0.9, desc="Guardando modelo...")

        # Save model as SavedModel format
        model_dir = f"./models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_dir, exist_ok=True)

        # Set proper permissions for the directory
        set_user_permissions(model_dir)

        model_saved_path = os.path.join(model_dir, "saved_model.tf")

        # When using MirroredStrategy, save model outside strategy scope
        with strategy.scope():
            model.save(model_saved_path, save_format='tf')

        # Set permissions for saved model files
        set_user_permissions(model_saved_path)
        
        # Save class names
        with open(f"{model_dir}/classes.txt", 'w') as f:
            f.write('\n'.join(class_names))
        
        # Save training history as JSON with reference script format
        accuracy_data = combined_history.get('accuracy', [])
        val_accuracy_data = combined_history.get('val_accuracy', [])
        loss_data = combined_history.get('loss', [])
        val_loss_data = combined_history.get('val_loss', [])
        
        history_data = {
            'epochs': list(range(1, len(accuracy_data) + 1)),
            # Keep original format for compatibility
            'train_acc': [float(acc * 100) for acc in accuracy_data],
            'val_acc': [float(acc * 100) for acc in val_accuracy_data],
            'train_loss': [float(loss) for loss in loss_data],
            'val_loss': [float(loss) for loss in val_loss_data],
            # Add reference script format - using only unfrozen phase (fine-tuning phase)
            'chart_unfrozen_accuracy': [float(acc) for acc in (unfrozen_history.get('accuracy', []) if unfrozen_history else [])],
            'chart_unfrozen_val_accuracy': [float(acc) for acc in (unfrozen_history.get('val_accuracy', []) if unfrozen_history else [])],
            'chart_unfrozen_loss': [float(loss) for loss in (unfrozen_history.get('loss', []) if unfrozen_history else [])],
            'chart_unfrozen_val_loss': [float(loss) for loss in (unfrozen_history.get('val_loss', []) if unfrozen_history else [])],
            # Add frozen phase for completeness
            'chart_frozen_accuracy': [float(acc) for acc in (frozen_history.get('accuracy', []) if frozen_history else [])],
            'chart_frozen_val_accuracy': [float(acc) for acc in (frozen_history.get('val_accuracy', []) if frozen_history else [])],
            'chart_frozen_loss': [float(loss) for loss in (frozen_history.get('loss', []) if frozen_history else [])],
            'chart_frozen_val_loss': [float(loss) for loss in (frozen_history.get('val_loss', []) if frozen_history else [])],
            # Training phases info
            'frozen_epochs': frozen_epochs,
            'unfrozen_epochs': unfrozen_epochs,
            'total_epochs': epochs
        }
        with open(f"{model_dir}/training_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save config
        config_data = {
            'model_name': model_name,
            'classes': class_names,
            'num_classes': len(class_names),
            'epochs': epochs,
            'frozen_learning_rate': frozen_learning_rate,
            'finetune_learning_rate': finetune_learning_rate,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{model_dir}/config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        current_model = model
        training_history = combined_history
        
        # Store training parameters globally for download functionality
        global current_training_params
        current_training_params = {
            'model_name': model_name,
            'img_size': img_size,
            'batch_size': batch_size,
            'epochs_total': epochs_total,
            'frozen_learning_rate': frozen_learning_rate,
            'finetune_learning_rate': finetune_learning_rate,
            'advanced_config': advanced_config,
            'augmentation_config': augmentation_config,
            'class_names': class_names.copy()
        }
        
        progress(1.0, desc="¡Entrenamiento completado!")
        
        # Create training plots
        plots_img = create_training_plots(frozen_history, unfrozen_history, combined_history)
        
        # Save training plots
        if plots_img:
            plots_img.save(f"{model_dir}/training_plots.png")

        # Set final permissions for all files in the model directory
        set_user_permissions(model_dir)

        # Training summary with error handling
        final_acc = combined_history.get('accuracy', [0])[-1] if combined_history.get('accuracy') else 0
        final_val_acc = combined_history.get('val_accuracy', [0])[-1] if combined_history.get('val_accuracy') else 0
        
        summary = f"""✅ **Entrenamiento Completado**

📊 **Resultados Finales:**
- **Precisión de entrenamiento:** {final_acc:.4f} ({final_acc*100:.2f}%)
- **Precisión de validación:** {final_val_acc:.4f} ({final_val_acc*100:.2f}%)
- **Épocas completadas:** {epochs}
- **Modelo:** {model_name}
- **Clases:** {len(class_names)}

⚙️ **Parámetros de Entrenamiento Utilizados:**
- **Modo de configuración:** {'Manual (Avanzado)' if advanced_config else 'Automático (Optimizado)'}
- **Modelo base:** {model_name}
- **Tamaño de imagen:** {img_size}x{img_size} px
- **Batch size:** {batch_size}{' (ajustado por tamaño de imagen)' if not advanced_config or batch_size != manual_batch_size else ''}
- **Learning rate fase congelada:** {frozen_learning_rate} (épocas 1-{frozen_epochs})
- **Learning rate fine-tuning:** {finetune_learning_rate} (épocas {frozen_epochs+1}-{epochs})
- **Épocas fase congelada:** {frozen_epochs}
- **Épocas fine-tuning:** {unfrozen_epochs}
- **Total de épocas:** {epochs}
- **Último epoch procesado:** {last_progress_epoch}
- **Número de clases:** {len(class_names)}
- **Clases:** {', '.join(class_names)}

💾 **Modelo guardado como SavedModel (.tf):**
- **Directorio:** `{model_dir}/`
- **Formato:** TensorFlow SavedModel
- **Ruta del modelo:** `{model_saved_path}`

🎯 **Próximos pasos:**
1. Revisa las métricas en la pestaña de visualización
2. Cuantiza el modelo para optimizar el rendimiento
3. Prueba el modelo con nuevas imágenes

¡Entrenamiento exitoso! 🎉"""

        # Restore original umask
        os.umask(old_umask)

        return summary, plots_img

    except Exception as e:
        # Restore original umask on error
        if 'old_umask' in locals():
            os.umask(old_umask)
        return f"❌ Error durante el entrenamiento: {str(e)}", None

def create_training_plots(frozen_history=None, unfrozen_history=None, combined_history=None):
    """Create training metrics plots with separated frozen/unfrozen phases"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Entrenamiento por Fases', fontsize=16, fontweight='bold')
        
        # Fallback: if no separated histories but combined_history exists, 
        # try to create a combined view or default message
        if not frozen_history and not unfrozen_history and combined_history:
            # Show combined data in all plots with a note
            epochs_range = range(1, len(combined_history['accuracy']) + 1)
            note_text = 'Datos combinados\n(sin separación de fases)'
            
            # All plots will show the same combined data with different focus
            for ax, title, ylabel in [(ax1, 'Accuracy - Fase Frozen', 'Accuracy'),
                                     (ax2, 'Accuracy - Fase Unfrozen', 'Accuracy'), 
                                     (ax3, 'Loss - Fase Frozen', 'Loss'),
                                     (ax4, 'Loss - Fase Unfrozen', 'Loss')]:
                if 'Accuracy' in ylabel:
                    ax.plot(epochs_range, combined_history['accuracy'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
                    ax.plot(epochs_range, combined_history['val_accuracy'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
                else:
                    ax.plot(epochs_range, combined_history['loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
                    ax.plot(epochs_range, combined_history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=8, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return Image.open(img_buffer)
        
        # Arriba izquierda: Accuracy frozen phase
        if frozen_history and 'accuracy' in frozen_history:
            frozen_epochs = range(1, len(frozen_history['accuracy']) + 1)
            ax1.plot(frozen_epochs, frozen_history['accuracy'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
            ax1.plot(frozen_epochs, frozen_history['val_accuracy'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
            ax1.set_title('Accuracy - Fase Frozen', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No hay datos\nde fase frozen', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Accuracy - Fase Frozen', fontweight='bold')
        
        # Arriba derecha: Accuracy unfrozen phase
        if unfrozen_history and 'accuracy' in unfrozen_history:
            unfrozen_epochs = range(1, len(unfrozen_history['accuracy']) + 1)
            ax2.plot(unfrozen_epochs, unfrozen_history['accuracy'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
            ax2.plot(unfrozen_epochs, unfrozen_history['val_accuracy'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
            ax2.set_title('Accuracy - Fase Unfrozen', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No hay datos\nde fase unfrozen', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Accuracy - Fase Unfrozen', fontweight='bold')
        
        # Abajo izquierda: Loss frozen phase
        if frozen_history and 'loss' in frozen_history:
            frozen_epochs = range(1, len(frozen_history['loss']) + 1)
            ax3.plot(frozen_epochs, frozen_history['loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
            ax3.plot(frozen_epochs, frozen_history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
            ax3.set_title('Loss - Fase Frozen', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No hay datos\nde fase frozen', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Loss - Fase Frozen', fontweight='bold')
        
        # Abajo derecha: Loss unfrozen phase
        if unfrozen_history and 'loss' in unfrozen_history:
            unfrozen_epochs = range(1, len(unfrozen_history['loss']) + 1)
            ax4.plot(unfrozen_epochs, unfrozen_history['loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
            ax4.plot(unfrozen_epochs, unfrozen_history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
            ax4.set_title('Loss - Fase Unfrozen', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No hay datos\nde fase unfrozen', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Loss - Fase Unfrozen', fontweight='bold')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return Image.open(img_buffer)
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        return None


def quantize_model(model_selection, quantization_types):
    

    # 1) Normaliza el valor del dropdown
    if isinstance(model_selection, (list, tuple)) and len(model_selection) >= 1:
        model_selection = model_selection[-1]  # coge la ruta/valor

    global current_model
    model_to_quantize = current_model
    model_name = "current_model"

    if model_selection and model_selection != "current":
        saved_model_path = f"{model_selection}/saved_model.tf"
        h5_model_path = f"{model_selection}/model.h5"
        try:
            if os.path.exists(saved_model_path):
                try:
                    model_to_quantize = tf.keras.models.load_model(saved_model_path)
                except Exception as e1:
                    # Reintentos compatibles (Normalization / versiones)
                    try:
                        model_to_quantize = tf.keras.models.load_model(saved_model_path, compile=False)
                    except Exception as e2:
                        try:
                            
                            custom_objects = {'Normalization': keras.layers.Normalization}
                            model_to_quantize = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects, compile=False)
                        except Exception as e3:
                            custom_objects = {'Normalization': tf.keras.layers.Normalization}
                            model_to_quantize = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects, compile=False)
            elif os.path.exists(h5_model_path):
                # Mismos reintentos para H5
                try:
                    model_to_quantize = tf.keras.models.load_model(h5_model_path)
                except Exception:
                    try:
                        model_to_quantize = tf.keras.models.load_model(h5_model_path, compile=False)
                    except Exception:
                        
                        custom_objects = {'Normalization': keras.layers.Normalization}
                        model_to_quantize = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objects, compile=False)
            else:
                return f"❌ No se encontró modelo en {model_selection}"
            model_name = os.path.basename(model_selection)
        except Exception as e:
            err = str(e)
            if "list index out of range" in err:
                return "❌ El modelo seleccionado no se pudo cargar (índices fuera de rango). Refresca la lista y vuelve a seleccionar, o reentrena para regenerar el SavedModel."
            return f"❌ Error cargando el modelo seleccionado: {err}"

    if model_to_quantize is None:
        return "❌ No hay modelo disponible. Entrena un modelo primero o selecciona uno entrenado."
    # ... resto de la función tal cual ...

    
    try:
        # Create export directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_export_dir = f"/workspace/models/quantized_{model_name}_{timestamp}"
        os.makedirs(local_export_dir, exist_ok=True)
        
        results = []
        total_size = 0
        
        # Define model filenames
        model_filename_float32 = f"{model_name}_float32.tflite"
        model_filename_float16 = f"{model_name}_float16.tflite"
        model_filename_int8 = f"{model_name}_int8.tflite"
        
        # Export paths
        export_path_float32 = os.path.join(local_export_dir, model_filename_float32)
        export_path_float16 = os.path.join(local_export_dir, model_filename_float16)
        export_path_int8 = os.path.join(local_export_dir, model_filename_int8)
        
        # Generate Float32 model (no quantization)
        if "float32" in quantization_types:
            try:
                converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model_to_quantize)
                tflite_model_f32 = converter_f32.convert()
                
                with open(export_path_float32, 'wb') as f:
                    f.write(tflite_model_f32)
                
                size_mb = len(tflite_model_f32) / 1024 / 1024
                total_size += size_mb
                results.append(f"✅ **Float32:** {size_mb:.2f} MB - `{model_filename_float32}`")
            except Exception as e:
                results.append(f"❌ **Float32:** Error - {str(e)}")
        
        # Generate Float16 model
        if "float16" in quantization_types:
            try:
                converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model_to_quantize)
                converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
                converter_f16.target_spec.supported_types = [tf.float16]
                tflite_model_f16 = converter_f16.convert()
                
                with open(export_path_float16, 'wb') as f:
                    f.write(tflite_model_f16)
                
                size_mb = len(tflite_model_f16) / 1024 / 1024
                total_size += size_mb
                results.append(f"✅ **Float16:** {size_mb:.2f} MB - `{model_filename_float16}`")
            except Exception as e:
                results.append(f"❌ **Float16:** Error - {str(e)}")
        
        # Generate Int8 model (Dynamic Range Quantization)
        if "int8" in quantization_types:
            try:
                converter_i8 = tf.lite.TFLiteConverter.from_keras_model(model_to_quantize)
                converter_i8.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model_i8 = converter_i8.convert()
                
                with open(export_path_int8, 'wb') as f:
                    f.write(tflite_model_i8)
                
                size_mb = len(tflite_model_i8) / 1024 / 1024
                total_size += size_mb
                results.append(f"✅ **Int8:** {size_mb:.2f} MB - `{model_filename_int8}`")
            except Exception as e:
                results.append(f"❌ **Int8:** Error - {str(e)}")
        
        # Create summary
        successful_exports = [r for r in results if r.startswith("✅")]
        
        result_text = f"""✅ **Cuantización Completada**

📊 **Modelo cuantizado:** {model_name}

🗂️ **Modelos generados:**
{chr(10).join(results)}

📈 **Resumen:**
- **Total de formatos:** {len(successful_exports)}/{len(quantization_types)}
- **Tamaño total:** {total_size:.2f} MB
- **Directorio:** `{local_export_dir}`

🎯 **Modelos optimizados listos para inferencia móvil!**
💡 **Recomendación:** Usa Int8 para mayor eficiencia, Float16 para balance, Float32 para máxima precisión."""
        
        return result_text
        
    except Exception as e:
        return f"❌ Error durante la cuantización: {str(e)}"

def get_available_trained_models():
    """Get list of available trained models"""
    models_dir = "/workspace/models"
    if not os.path.exists(models_dir):
        return []

    models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            # Check for SavedModel format
            saved_model_path = os.path.join(model_path, "saved_model.tf")
            # Also check for legacy h5 format for backward compatibility
            h5_model_path = os.path.join(model_path, "model.h5")

            if os.path.exists(saved_model_path) or os.path.exists(h5_model_path):
                models.append((item, model_path))

    return models

def get_available_formats_for_model(model_path):
    """Get list of available formats for a specific model"""
    if not model_path or model_path == "current":
        return ["SavedModel (Actual en memoria)"]

    formats = []

    # Check for SavedModel format
    if os.path.exists(os.path.join(model_path, "saved_model.tf")):
        formats.append("SavedModel (.tf)")

    # Check for H5 format
    if os.path.exists(os.path.join(model_path, "model.h5")):
        formats.append("H5 (.h5)")

    # Check for quantized models in the same directory
    model_dir = model_path
    if os.path.isdir(model_dir):
        for item in os.listdir(model_dir):
            if item.endswith("_float32.tflite"):
                formats.append("TFLite Float32")
            elif item.endswith("_float16.tflite"):
                formats.append("TFLite Float16")
            elif item.endswith("_int8.tflite"):
                formats.append("TFLite Int8")

    # Also check for quantized folders
    models_dir = "/workspace/models"
    if os.path.exists(models_dir):
        model_basename = os.path.basename(model_path)
        for item in os.listdir(models_dir):
            if item.startswith(f"quantized_{model_basename}"):
                quantized_path = os.path.join(models_dir, item)
                if os.path.isdir(quantized_path):
                    for qfile in os.listdir(quantized_path):
                        if qfile.endswith("_float32.tflite") and "TFLite Float32" not in formats:
                            formats.append("TFLite Float32")
                        elif qfile.endswith("_float16.tflite") and "TFLite Float16" not in formats:
                            formats.append("TFLite Float16")
                        elif qfile.endswith("_int8.tflite") and "TFLite Int8" not in formats:
                            formats.append("TFLite Int8")

    if not formats:
        formats.append("SavedModel (.tf)")

    return formats

def load_training_plots(model_dir=None):
    """Load training plots and metrics for a specific model or current model"""
    if model_dir:
        # Load specific model's plots and metrics
        plots_path = f"{model_dir}/training_plots.png"
        history_path = f"{model_dir}/training_history.json"
        config_path = f"{model_dir}/config.json"
        
        plots = None
        if os.path.exists(plots_path):
            try:
                plots = Image.open(plots_path)
            except:
                plots = None
        
        if plots is None:
            return None, "❌ No hay gráficas disponibles para este modelo."
        
        summary = f"📊 **Resumen del Entrenamiento**\n\n"
        
        # Load model config for dataset info
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get('model_name', 'Desconocido')
                    classes = config.get('classes', [])
                    summary += f"• **Modelo**: {model_name}\n"
                    summary += f"• **Dataset**: {len(classes)} clases\n"
                    summary += f"• **Clases**: {', '.join(classes)}\n\n"
            except:
                pass
        
        # Load training metrics
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    metrics = json.load(f)
                    epochs = len(metrics.get('epochs', []))
                    
                    # Try to use reference script format first
                    unfrozen_acc = metrics.get('chart_unfrozen_accuracy', [])
                    unfrozen_val_acc = metrics.get('chart_unfrozen_val_accuracy', [])
                    unfrozen_loss = metrics.get('chart_unfrozen_loss', [])
                    unfrozen_val_loss = metrics.get('chart_unfrozen_val_loss', [])
                    
                    # Fallback to original format if reference format not available
                    if not unfrozen_acc:
                        train_acc = metrics.get('train_acc', [])
                        val_acc = metrics.get('val_acc', [])
                        train_loss = metrics.get('train_loss', [])
                        val_loss = metrics.get('val_loss', [])
                        
                        # Convert percentages back to decimals if needed
                        unfrozen_acc = [acc/100 if acc > 1 else acc for acc in train_acc]
                        unfrozen_val_acc = [acc/100 if acc > 1 else acc for acc in val_acc]
                        unfrozen_loss = train_loss
                        unfrozen_val_loss = val_loss
                    
                    summary += f"📈 **Épocas totales:** {epochs}\n"
                    
                    # Get training phases info
                    frozen_epochs = metrics.get('frozen_epochs', 0)
                    unfrozen_epochs = metrics.get('unfrozen_epochs', epochs)
                    
                    if frozen_epochs > 0:
                        summary += f"🔒 **Fase frozen:** {frozen_epochs} épocas\n"
                        summary += f"🔓 **Fase unfrozen:** {unfrozen_epochs} épocas\n\n"
                    
                    if unfrozen_acc and unfrozen_val_acc:
                        summary += f"🎯 **Mejor precisión entrenamiento:** {max(unfrozen_acc)*100:.2f}%\n"
                        summary += f"🎯 **Mejor precisión validación:** {max(unfrozen_val_acc)*100:.2f}%\n"
                        summary += f"📉 **Menor pérdida entrenamiento:** {min(unfrozen_loss):.4f}\n"
                        summary += f"📉 **Menor pérdida validación:** {min(unfrozen_val_loss):.4f}\n"
                        
                        # Check for overfitting using unfrozen phase
                        final_train_acc = unfrozen_acc[-1] * 100
                        final_val_acc = unfrozen_val_acc[-1] * 100
                        if final_train_acc - final_val_acc > 10:
                            summary += f"\n⚠️ **Advertencia**: Posible sobreajuste (diferencia: {final_train_acc - final_val_acc:.1f}%)"
            except Exception as e:
                summary += f"• Error cargando métricas de entrenamiento: {str(e)}\n"
        
        return plots, summary
    else:
        # Default behavior for current model
        global training_history
        if training_history is None:
            return None, "❌ No hay gráficas disponibles. Entrena un modelo primero."
        
        plots = create_training_plots(combined_history=training_history)
        if plots is None:
            return None, "❌ No hay gráficas disponibles. Entrena un modelo primero."
        
        summary = f"📊 **Resumen del Entrenamiento**\n\n"
        summary += f"📈 **Épocas totales:** {len(training_history['accuracy'])}\n"
        summary += f"🎯 **Mejor precisión entrenamiento:** {max(training_history['accuracy']):.2f}%\n"
        summary += f"🎯 **Mejor precisión validación:** {max(training_history['val_accuracy']):.2f}%\n"
        summary += f"📉 **Menor pérdida entrenamiento:** {min(training_history['loss']):.4f}\n"
        summary += f"📉 **Menor pérdida validación:** {min(training_history['val_loss']):.4f}\n"
        
        # Check for overfitting
        final_train_acc = training_history['accuracy'][-1]
        final_val_acc = training_history['val_accuracy'][-1]
        if final_train_acc - final_val_acc > 10:
            summary += f"\n⚠️ **Advertencia:** Posible sobreajuste detectado (diferencia: {final_train_acc - final_val_acc:.1f}%)"
        
        return plots, summary

def perform_inference(image, model_path=None, model_format=None):
    """Perform inference on uploaded image with support for different formats"""
    global current_model, class_names

    model_to_use = current_model
    classes_to_use = class_names
    using_tflite = False
    tflite_interpreter = None
    
    # Load specific model if path is provided
    if model_path and model_path != "current":
        try:
            # Check if TFLite format is requested
            if model_format and "TFLite" in model_format:
                # Determine TFLite file suffix based on format
                if "Float32" in model_format:
                    suffix = "_float32.tflite"
                elif "Float16" in model_format:
                    suffix = "_float16.tflite"
                elif "Int8" in model_format:
                    suffix = "_int8.tflite"
                else:
                    return f"❌ Formato TFLite no reconocido: {model_format}"

                # Search for TFLite file in model directory
                tflite_path = None
                model_basename = os.path.basename(model_path)

                # Check in model directory
                if os.path.isdir(model_path):
                    for item in os.listdir(model_path):
                        if item.endswith(suffix):
                            tflite_path = os.path.join(model_path, item)
                            break

                # Check in quantized folders
                if not tflite_path:
                    models_dir = "/workspace/models"
                    if os.path.exists(models_dir):
                        for item in os.listdir(models_dir):
                            if item.startswith(f"quantized_{model_basename}"):
                                quantized_path = os.path.join(models_dir, item)
                                if os.path.isdir(quantized_path):
                                    for qfile in os.listdir(quantized_path):
                                        if qfile.endswith(suffix):
                                            tflite_path = os.path.join(quantized_path, qfile)
                                            break
                                if tflite_path:
                                    break

                if not tflite_path or not os.path.exists(tflite_path):
                    return f"❌ No se encontró archivo TFLite con formato {model_format} para el modelo seleccionado"

                # Load TFLite model
                tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
                tflite_interpreter.allocate_tensors()
                using_tflite = True

            else:
                # Load SavedModel or H5 format
                saved_model_path = f"{model_path}/saved_model.tf"
                h5_model_path = f"{model_path}/model.h5"

                if os.path.exists(saved_model_path):
                    # Load with custom object scope to handle version compatibility
                    try:
                        model_to_use = tf.keras.models.load_model(saved_model_path)
                    except Exception as e:
                        # Handle EfficientNet Normalization layer compatibility issues
                        if "Normalization" in str(e):
                            # Try loading without compiling first
                            try:
                                model_to_use = tf.keras.models.load_model(saved_model_path, compile=False)
                            except Exception:
                                # Use custom object scope with keras.layers.Normalization
                                import keras
                                custom_objects = {
                                    'Normalization': keras.layers.Normalization
                                }
                                try:
                                    model_to_use = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects, compile=False)
                                except Exception:
                                    # Last resort: try with tf.keras namespace
                                    custom_objects = {
                                        'Normalization': tf.keras.layers.Normalization
                                    }
                                    try:
                                        model_to_use = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects, compile=False)
                                    except Exception:
                                        # If all attempts fail, provide helpful message
                                        model_name = os.path.basename(model_path)
                                        return f"❌ El modelo {model_name} no es compatible con TensorFlow 2.9.1. Este modelo fue entrenado con una versión diferente de TensorFlow que usa una capa Normalization incompatible. Solución: entrena un nuevo modelo con la versión actual o usa un modelo ResNet que es más compatible entre versiones."
                        else:
                            raise e
                elif os.path.exists(h5_model_path):
                    # Fallback to h5 format for backward compatibility
                    try:
                        model_to_use = tf.keras.models.load_model(h5_model_path)
                    except Exception as e:
                        # Handle EfficientNet Normalization layer compatibility issues
                        if "Normalization" in str(e):
                            # Try loading without compiling first
                            try:
                                model_to_use = tf.keras.models.load_model(h5_model_path, compile=False)
                            except Exception:
                                # Use custom object scope with keras.layers.Normalization
                                import keras
                                custom_objects = {
                                    'Normalization': keras.layers.Normalization
                                }
                                try:
                                    model_to_use = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objects, compile=False)
                                except Exception:
                                    # Last resort: try with tf.keras namespace
                                    custom_objects = {
                                        'Normalization': tf.keras.layers.Normalization
                                    }
                                    try:
                                        model_to_use = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objects, compile=False)
                                    except Exception:
                                        # If all attempts fail, provide helpful message
                                        model_name = os.path.basename(model_path)
                                        return f"❌ El modelo {model_name} no es compatible con TensorFlow 2.9.1. Este modelo fue entrenado con una versión diferente de TensorFlow que usa una capa Normalization incompatible. Solución: entrena un nuevo modelo con la versión actual o usa un modelo ResNet que es más compatible entre versiones."
                        else:
                            raise e
                else:
                    return f"❌ No se encontró modelo en {model_path}"

            # Load class names for this model (check both labels.txt and classes.txt for compatibility)
            labels_file = f"{model_path}/labels.txt"
            classes_file = f"{model_path}/classes.txt"

            if os.path.exists(labels_file):
                with open(labels_file, 'r') as f:
                    classes_to_use = [line.strip() for line in f.readlines()]
            elif os.path.exists(classes_file):
                with open(classes_file, 'r') as f:
                    classes_to_use = [line.strip() for line in f.readlines()]
            else:
                classes_to_use = class_names
        except Exception as e:
            error_msg = str(e)
            if "list index out of range" in error_msg:
                model_name = os.path.basename(model_path) if model_path else "seleccionado"
                return f"❌ El modelo {model_name} tiene una configuración corrupta. El archivo SavedModel está dañado y debe ser reentrenado."
            elif "Normalization" in error_msg:
                model_name = os.path.basename(model_path) if model_path else "seleccionado"
                return f"❌ El modelo {model_name} no es compatible con TensorFlow 2.9.1. Entrena un nuevo modelo con la versión actual o usa ResNet para mejor compatibilidad."
            else:
                return f"❌ Error cargando el modelo seleccionado: {error_msg}"
    
    if model_to_use is None and tflite_interpreter is None:
        return "❌ No hay modelo disponible. Entrena un modelo primero o selecciona uno entrenado."

    if image is None:
        return "❌ No se ha subido ninguna imagen."

    try:
        if using_tflite and tflite_interpreter:
            # TFLite inference
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            # Get input size
            input_shape = input_details[0]['shape']
            img_size = input_shape[1]

            # Preprocess image
            img_array = np.array(image.resize((img_size, img_size)))
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32) / 255.0

            # Set input tensor
            tflite_interpreter.set_tensor(input_details[0]['index'], img_array)

            # Run inference
            tflite_interpreter.invoke()

            # Get output tensor
            predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]

        else:
            # Keras model inference
            # Get input size from model
            input_shape = model_to_use.input.shape
            if len(input_shape) < 2:
                return f"❌ Formato de modelo incompatible. Input shape: {input_shape}"
            img_size = input_shape[1] if len(input_shape) > 1 else 224  # Default to 224 if shape is unexpected

            # Preprocess image
            img_array = np.array(image.resize((img_size, img_size)))
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32) / 255.0

            # Make prediction
            predictions = model_to_use.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
        
        # Check if number of predictions matches number of classes
        if len(predictions[0]) != len(classes_to_use):
            return f"❌ Incompatibilidad de clases. Modelo predice {len(predictions[0])} clases, pero hay {len(classes_to_use)} etiquetas."
        
        # Check if predicted_class_idx is within bounds
        if predicted_class_idx >= len(classes_to_use):
            return f"❌ Índice de predicción fuera de rango: {predicted_class_idx} >= {len(classes_to_use)}"
        
        # Format results
        results = []
        for i, (class_name, prob) in enumerate(zip(classes_to_use, predictions[0])):
            status = "🏆" if i == predicted_class_idx else "   "
            results.append(f"{status} {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Generate model info string
        if using_tflite:
            model_info = f"TFLite ({model_format})"
        else:
            model_info = f"{model_to_use.name} (SavedModel/Keras)"

        prediction_text = f"""🔮 **Resultado de Predicción**

🎯 **Predicción Principal:**
**{classes_to_use[predicted_class_idx]}** - {confidence:.4f} ({confidence*100:.2f}%)

📊 **Todas las Probabilidades:**
{chr(10).join(results)}

ℹ️ **Información:**
- **Modelo utilizado:** {model_info}
- **Formato:** {model_format if model_format else 'SavedModel (memoria)'}
- **Clases disponibles:** {len(classes_to_use)}
- **Confianza:** {'Alta' if confidence > 0.8 else 'Media' if confidence > 0.5 else 'Baja'}

{'✅ Predicción confiable' if confidence > 0.7 else '⚠️ Predicción incierta' if confidence > 0.5 else '❌ Predicción poco confiable'}"""
        
        return prediction_text
        
    except Exception as e:
        return f"❌ Error durante la inferencia: {str(e)}"

def create_download_package(model_name, img_size, batch_size, epochs_total, frozen_learning_rate, finetune_learning_rate, 
                           advanced_config, augmentation_config, class_names_list):
    """Create a downloadable package with model and hyperparameters"""
    global current_model
    
    if current_model is None:
        return None, "❌ No hay modelo entrenado disponible para descargar"
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        package_name = f"{model_name}_{timestamp}"
        
        # Create temporary directory for the package
        temp_package_dir = tempfile.mkdtemp()
        package_path = os.path.join(temp_package_dir, package_name)
        os.makedirs(package_path, exist_ok=True)
        
        # Save the model in SavedModel format
        model_path = os.path.join(package_path, "saved_model")
        current_model.save(model_path, save_format='tf')

        # Generate TensorFlow Lite model
        tflite_model_path = os.path.join(package_path, "model.tflite")
        try:
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(current_model)

            # Optional: Enable optimization for size reduction
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Convert the model
            tflite_model = converter.convert()

            # Save the TensorFlow Lite model
            with open(tflite_model_path, "wb") as f:
                f.write(tflite_model)

        except Exception as e:
            print(f"Warning: Could not generate TensorFlow Lite model: {e}")
            # Create a placeholder file to indicate the error
            with open(os.path.join(package_path, "tflite_error.txt"), "w") as f:
                f.write(f"TensorFlow Lite conversion failed: {e}\n")
                f.write("You can manually convert the SavedModel using:\n")
                f.write("python -c \"import tensorflow as tf; converter = tf.lite.TFLiteConverter.from_saved_model('saved_model'); tflite_model = converter.convert(); open('model.tflite', 'wb').write(tflite_model)\"")
            tflite_model_path = None
        
        # Create hyperparameters JSON
        hyperparams = {
            "epochs": epochs_total,
            "batch_size": batch_size,
            "learning_rate_frozen": frozen_learning_rate,
            "learning_rate_unfrozen": finetune_learning_rate,
            "model": model_name,
            "img_size": img_size,
            "data_augmentation": augmentation_config,
            "advanced_config_used": advanced_config,
            "num_classes": len(class_names_list),
            "classes": class_names_list,
            "timestamp": timestamp,
            "framework": "TensorFlow",
            "model_format": "SavedModel + TensorFlow Lite",
            "tflite_included": tflite_model_path is not None
        }
        
        hyperparams_path = os.path.join(package_path, "hyperparams.json")
        with open(hyperparams_path, "w", encoding='utf-8') as f:
            json.dump(hyperparams, f, indent=2, ensure_ascii=False)
        
        # Create labels.txt for compatibility
        labels_path = os.path.join(package_path, "labels.txt")
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(class_names_list))
        
        # Create README with instructions
        readme_path = os.path.join(package_path, "README.md")
        tflite_content = ""
        if tflite_model_path is not None:
            tflite_content = """
## Uso del modelo TensorFlow Lite (model.tflite):
```python
import tensorflow as tf
import numpy as np

# Cargar el modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Obtener detalles de input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preparar datos de entrada (ejemplo)
# input_data debe ser un numpy array con shape y dtype correctos
input_data = np.array(your_input_data, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Ejecutar inferencia
interpreter.invoke()

# Obtener resultados
output_data = interpreter.get_tensor(output_details[0]['index'])
```
"""
        else:
            tflite_content = """
## TensorFlow Lite:
⚠️ El modelo TensorFlow Lite no pudo ser generado automáticamente.
Ver el archivo `tflite_error.txt` para más detalles sobre cómo convertir manualmente.
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Modelo {model_name} - {timestamp}

## Contenido del paquete:
- `saved_model/`: Modelo en formato TensorFlow SavedModel
- `model.tflite`: Modelo optimizado para TensorFlow Lite (si disponible)
- `hyperparams.json`: Hiperparámetros de entrenamiento
- `labels.txt`: Lista de etiquetas del dataset
- `README.md`: Este archivo

## Información del modelo:
- **Arquitectura**: {model_name}
- **Clases**: {len(class_names_list)} ({', '.join(class_names_list)})
- **Tamaño de imagen**: {img_size}x{img_size} px
- **Épocas**: {epochs_total}
- **Batch size**: {batch_size}

## Cómo cargar el modelo SavedModel:
```python
import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model('saved_model')

# Ver resumen del modelo
model.summary()
```
{tflite_content}
## Hiperparámetros:
Ver `hyperparams.json` para todos los detalles de entrenamiento.

## Etiquetas de clasificación:
Ver `labels.txt` para la lista completa de clases que el modelo puede predecir.
""")
        
        # Create ZIP file
        zip_path = os.path.join(temp_package_dir, f"{package_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_package_dir)
                    zipf.write(file_path, arc_name)
        
        return zip_path, f"✅ Paquete creado: {package_name}.zip"
        
    except Exception as e:
        return None, f"❌ Error creando el paquete: {str(e)}"

def download_current_model():
    """Download the currently trained model with hyperparameters"""
    global current_training_params
    
    if not current_training_params:
        return None, "❌ No hay modelo entrenado disponible. Entrena un modelo primero."
    
    return create_download_package(
        current_training_params['model_name'],
        current_training_params['img_size'], 
        current_training_params['batch_size'],
        current_training_params['epochs_total'],
        current_training_params['frozen_learning_rate'],
        current_training_params['finetune_learning_rate'],
        current_training_params['advanced_config'],
        current_training_params['augmentation_config'],
        current_training_params['class_names']
    )

# Gradio Interface with Rachael.vision theme
with gr.Blocks(title=t('title'), theme=gr.themes.Soft().set(
    background_fill_primary='#f8f9fa',
    background_fill_secondary='#e9ecef', 
    block_background_fill='#ffffff',
    body_text_color='#212529',
    block_label_text_color='#495057',
    button_primary_background_fill='#0d6efd',
    button_primary_background_fill_hover='#0b5ed7',
    button_secondary_background_fill='#6c757d',
    button_secondary_background_fill_hover='#5c636a'
)) as app:
    
    gr.Markdown(f"# {t('title')}")
    gr.Markdown(f"### {t('subtitle_professional')}")
    gr.Markdown(t('subtitle_advanced'))
    
    # Dataset Validation Tab
    with gr.Tab(f"📁 {t('dataset_tab')}"):
        gr.Markdown(f"### {t('upload_validate_title')}")
        gr.Markdown(t('upload_validate_desc'))
        
        zip_input = gr.File(label=t('upload_zip'), file_types=[".zip"])
        validate_btn = gr.Button(f"🔍 {t('validate_button')}", variant="primary")
        validation_output = gr.Textbox(label=f"✅ {t('validation_result')}", lines=15)
        
        validate_btn.click(validate_dataset, inputs=[zip_input], outputs=[validation_output])
    
    # Data Augmentation Tab
    with gr.Tab(f"🎨 {t('augmentation_tab')}"):
        gr.Markdown(f"### {t('augmentation_title')}")
        gr.Markdown(t('augmentation_desc'))
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🔄 Rotación")
                rotation_enabled = gr.Checkbox(label="Activar rotación", value=True)
                rotation_degrees = gr.Slider(
                    minimum=0, maximum=180, value=15, step=5,
                    label="Grados de rotación (0° - 180°)",
                    info="Rotación máxima en grados"
                )
                
                gr.Markdown("#### ↔️ Traslación")
                translation_enabled = gr.Checkbox(label="Activar traslación", value=True)
                translation_x_factor = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                    label="Factor de traslación X (0% - 100%)",
                    info="Porcentaje de desplazamiento horizontal"
                )
                translation_y_factor = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                    label="Factor de traslación Y (0% - 100%)",
                    info="Porcentaje de desplazamiento vertical"
                )
                
                gr.Markdown("#### 🔄 Volteos")
                flip_horizontal = gr.Checkbox(label="Volteo horizontal", value=True)
                flip_vertical = gr.Checkbox(label="Volteo vertical", value=False)
                
            with gr.Column():
                gr.Markdown("#### ☀️ Brightness")
                brightness_enabled = gr.Checkbox(label="Activar ajustes de brillo", value=False)
                brighten_factor = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    label="Brighten (0% - 100%)",
                    info="Incremento de brillo"
                )
                darken_factor = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    label="Darken (0% - 100%)",
                    info="Reducción de brillo"
                )
                
                gr.Markdown("#### 🔍 Zoom")
                zoom_enabled = gr.Checkbox(label="Activar zoom", value=True)
                zoom_in_factor = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                    label="Zoom In (0% - 100%)",
                    info="Factor de acercamiento"
                )
                zoom_out_factor = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                    label="Zoom Out (0% - 100%)",
                    info="Factor de alejamiento"
                )
        
        # Data Augmentation preview and confirmation
        gr.Markdown("---")
        with gr.Row():
            preview_augmentation_btn = gr.Button("🔍 Vista Previa de Configuración", variant="secondary")
            preview_visual_btn = gr.Button("🖼️ Vista Previa Visual (3x3)", variant="secondary")
            confirm_augmentation_btn = gr.Button("✅ Confirmar Data Augmentation", variant="primary")

        augmentation_preview = gr.Textbox(
            label="📊 Configuración de Data Augmentation",
            lines=8,
            interactive=False
        )

        augmentation_visual_preview = gr.Image(
            label="🎨 Vista Previa Visual - Grid 3x3 con Augmentaciones Aplicadas",
            type="pil",
            interactive=False
        )
        
        # Add callback functions for the buttons
        def generate_preview():
            return preview_augmentation_config(
                rotation_enabled.value if hasattr(rotation_enabled, 'value') else False,
                rotation_degrees.value if hasattr(rotation_degrees, 'value') else 0,
                translation_enabled.value if hasattr(translation_enabled, 'value') else False,
                translation_x_factor.value if hasattr(translation_x_factor, 'value') else 0,
                translation_y_factor.value if hasattr(translation_y_factor, 'value') else 0,
                flip_horizontal.value if hasattr(flip_horizontal, 'value') else False,
                flip_vertical.value if hasattr(flip_vertical, 'value') else False,
                zoom_enabled.value if hasattr(zoom_enabled, 'value') else False,
                zoom_in_factor.value if hasattr(zoom_in_factor, 'value') else 0,
                zoom_out_factor.value if hasattr(zoom_out_factor, 'value') else 0,
                brightness_enabled.value if hasattr(brightness_enabled, 'value') else False,
                brighten_factor.value if hasattr(brighten_factor, 'value') else 0,
                darken_factor.value if hasattr(darken_factor, 'value') else 0
            )
        
        def confirm_augmentation():
            preview = generate_preview()
            confirmed_message = preview + "\n\n✅ **CONFIRMADO**: Configuración guardada para entrenamiento"
            return confirmed_message
        
        # Connect the buttons
        preview_augmentation_btn.click(
            lambda rot_en, rot_deg, trans_en, trans_x, trans_y, flip_h, flip_v, zoom_en, zoom_in, zoom_out, bright_en, brighten, darken:
            preview_augmentation_config(rot_en, rot_deg, trans_en, trans_x, trans_y, flip_h, flip_v, zoom_en, zoom_in, zoom_out, bright_en, brighten, darken),
            inputs=[
                rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                flip_horizontal, flip_vertical, zoom_enabled, zoom_in_factor, zoom_out_factor,
                brightness_enabled, brighten_factor, darken_factor
            ],
            outputs=augmentation_preview
        )

        preview_visual_btn.click(
            lambda rot_en, rot_deg, trans_en, trans_x, trans_y, flip_h, flip_v, zoom_en, zoom_in, zoom_out, bright_en, brighten, darken:
            generate_augmentation_preview_grid(rot_en, rot_deg, trans_en, trans_x, trans_y, flip_h, flip_v, zoom_en, zoom_in, zoom_out, bright_en, brighten, darken),
            inputs=[
                rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                flip_horizontal, flip_vertical, zoom_enabled, zoom_in_factor, zoom_out_factor,
                brightness_enabled, brighten_factor, darken_factor
            ],
            outputs=augmentation_visual_preview
        )

        confirm_augmentation_btn.click(
            lambda rot_en, rot_deg, trans_en, trans_x, trans_y, flip_h, flip_v, zoom_en, zoom_in, zoom_out, bright_en, brighten, darken:
            preview_augmentation_config(rot_en, rot_deg, trans_en, trans_x, trans_y, flip_h, flip_v, zoom_en, zoom_in, zoom_out, bright_en, brighten, darken) + "\n\n✅ **CONFIRMADO**: Configuración guardada para entrenamiento",
            inputs=[
                rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                flip_horizontal, flip_vertical, zoom_enabled, zoom_in_factor, zoom_out_factor,
                brightness_enabled, brighten_factor, darken_factor
            ],
            outputs=augmentation_preview
        )
    
    # Training Tab
    with gr.Tab(f"🚀 {t('training_tab')}"):
        gr.Markdown(f"### {t('training_title')}")
        gr.Markdown(t('training_desc'))
        
        with gr.Row():
            available_choices = get_model_choices()
            # Use ResNet50 as default (guaranteed to be available in TF 2.8)
            default_value = 'ResNet50' if 'ResNet50' in available_choices else available_choices[0] if available_choices else 'ResNet50'
            
            model_dropdown = gr.Dropdown(
                choices=available_choices,
                label=t('model_selection'),
                value=default_value
            )
        
        # Advanced configuration toggle
        advanced_config = gr.Checkbox(
            label="⚙️ Configuración Avanzada", 
            value=False,
            info="Permitir control manual de parámetros de entrenamiento"
        )
        
        # Default configuration display (shown when advanced is disabled)
        with gr.Group(visible=True) as default_config_group:
            gr.Markdown("### 📋 **Configuración Automática Optimizada**")
            default_info = gr.Markdown("""
            🔸 **Épocas:** Definidas por variable EPOCHS (40 por defecto)  
            🔸 **Learning Rate:** 0.01 (frozen) → 0.0001 (fine-tuning)  
            🔸 **Batch Size:** Dinámico basado en tamaño de imagen (64/32/16)  
            
            *Configuración optimizada para mejores resultados en transfer learning*
            """)
        
        # Advanced configuration controls (hidden by default)
        with gr.Group(visible=False) as advanced_config_group:
            gr.Markdown("### ⚙️ **Configuración Manual**")
            with gr.Row():
                epochs_input = gr.Slider(5, 100, value=40, step=1, label="Épocas Totales")
                batch_size_input = gr.Slider(8, 64, value=64, step=8, label="Batch Size Base")
            with gr.Row():
                frozen_lr_input = gr.Number(value=0.01, label="Learning Rate (Frozen)", precision=6)
                unfrozen_lr_input = gr.Number(value=0.0001, label="Learning Rate (Fine-tuning)", precision=6)
        
        # Default values when not using advanced config (will be passed but ignored)
        epochs_default = gr.State(40)
        batch_size_default = gr.State(64) 
        frozen_lr_default = gr.State(0.01)
        unfrozen_lr_default = gr.State(0.0001)
        
        train_btn = gr.Button(f"🚀 {t('start_training')}", variant="primary")
        
        # Download section
        gr.Markdown("---")
        gr.Markdown("### 📦 **Descargar Modelo Entrenado**")
        with gr.Row():
            download_btn = gr.Button("💾 Descargar Modelo + Hiperparámetros", variant="secondary")
            download_file = gr.File(label="📁 Archivo de Descarga", visible=False)
        download_status = gr.Textbox(label="Estado de la Descarga", lines=2, visible=False)
        
        training_output = gr.Textbox(label=t('training_output'), lines=15)
        training_plots = gr.Image(label="📊 Gráficas de Entrenamiento")
        
        # Toggle visibility of configuration groups
        def toggle_config_visibility(is_advanced):
            return {
                default_config_group: gr.Group(visible=not is_advanced),
                advanced_config_group: gr.Group(visible=is_advanced)
            }
        
        advanced_config.change(
            toggle_config_visibility,
            inputs=[advanced_config],
            outputs=[default_config_group, advanced_config_group]
        )
        
        # Download functionality
        def handle_download():
            zip_path, status = download_current_model()
            if zip_path:
                return {
                    download_file: gr.File(value=zip_path, visible=True),
                    download_status: gr.Textbox(value=status, visible=True)
                }
            else:
                return {
                    download_file: gr.File(visible=False),
                    download_status: gr.Textbox(value=status, visible=True)
                }
        
        download_btn.click(
            handle_download,
            outputs=[download_file, download_status]
        )
        
        train_btn.click(
            train_model,
            inputs=[
                model_dropdown, advanced_config, epochs_input, batch_size_input, frozen_lr_input, unfrozen_lr_input,
                rotation_enabled, rotation_degrees, translation_enabled, translation_x_factor, translation_y_factor,
                flip_horizontal, flip_vertical, zoom_enabled, zoom_in_factor, zoom_out_factor,
                brightness_enabled, brighten_factor, darken_factor
            ],
            outputs=[training_output, training_plots],
            show_progress="minimal"
        )
    
    # Metrics Visualization Tab
    with gr.Tab(f"📊 {t('metrics_tab')}"):
        gr.Markdown(f"### {t('metrics_title')}")
        gr.Markdown(t('metrics_desc'))
        
        with gr.Row():
            metrics_model_dropdown = gr.Dropdown(
                choices=[],
                label="📋 Modelo Entrenado",
                info="Selecciona un modelo para ver sus métricas"
            )
            refresh_metrics_btn = gr.Button(f"🔄 {t('refresh_button')}")
        
        load_metrics_btn = gr.Button(f"📈 {t('load_metrics')}", variant="secondary")
        metrics_display = gr.Image(label=t('metrics_plot'))
        metrics_summary = gr.Textbox(label="📊 Resumen de Métricas", lines=8)
        
        def refresh_metrics_models():
            models = get_available_trained_models()
            choices = [("Modelo Actual", None)] + models
            return gr.Dropdown(choices=choices)
        
        def load_selected_metrics(model_selection):
            if model_selection is None:
                return load_training_plots()
            else:
                return load_training_plots(model_selection)
        
        refresh_metrics_btn.click(
            refresh_metrics_models,
            outputs=metrics_model_dropdown
        )
        
        load_metrics_btn.click(
            load_selected_metrics,
            inputs=metrics_model_dropdown,
            outputs=[metrics_display, metrics_summary]
        )
    
    # Quantization Tab
    with gr.Tab(f"⚡ {t('quantization_tab')}"):
        gr.Markdown(f"### {t('quantization_title')}")
        gr.Markdown(t('quantization_desc'))
        
        with gr.Row():
            quant_model_dropdown = gr.Dropdown(
                choices=[],
                label="🤖 Modelo para Cuantizar",
                info="Selecciona el modelo a cuantizar"
            )
            refresh_quant_btn = gr.Button(f"🔄 {t('refresh_button')}")
        
        gr.Markdown("#### 📊 Formatos de Cuantización")
        gr.Markdown("Selecciona los formatos que deseas generar:")
        
        with gr.Row():
            with gr.Column():
                float32_check = gr.Checkbox(label="🔹 Float32", value=True, info="Precisión completa (más grande)")
                float16_check = gr.Checkbox(label="🔸 Float16", value=True, info="Precisión media (balance)")
            with gr.Column():
                int8_check = gr.Checkbox(label="🔺 Int8", value=True, info="Máxima compresión (más pequeño)")
        
        quantize_btn = gr.Button(f"⚡ {t('quantize_model')}", variant="primary")
        quantization_output = gr.Textbox(label=t('quantization_output'), lines=15)
        
        def refresh_quantization_models():
            models = get_available_trained_models()
            choices = [("Modelo Actual", "current")] + models
            return gr.Dropdown(choices=choices)
        
        def quantize_selected_model(model_selection, float32, float16, int8):
            quantization_types = []
            if float32:
                quantization_types.append("float32")
            if float16:
                quantization_types.append("float16")
            if int8:
                quantization_types.append("int8")
            
            if not quantization_types:
                return "❌ Selecciona al menos un formato de cuantización"
                
            return quantize_model(model_selection, quantization_types)
        
        refresh_quant_btn.click(
            refresh_quantization_models,
            outputs=quant_model_dropdown
        )
        
        quantize_btn.click(
            quantize_selected_model, 
            inputs=[quant_model_dropdown, float32_check, float16_check, int8_check], 
            outputs=[quantization_output]
        )
    
    # Inference Tab
    with gr.Tab(f"🔮 {t('inference_tab')}"):
        gr.Markdown(f"### {t('inference_title')}")
        gr.Markdown(t('inference_desc'))

        with gr.Row():
            prediction_model_dropdown = gr.Dropdown(
                choices=[],
                label="🤖 Modelo para Predicción",
                info="Selecciona el modelo a usar para la inferencia"
            )
            refresh_prediction_btn = gr.Button(f"🔄 {t('refresh_button')}")

        with gr.Row():
            prediction_format_dropdown = gr.Dropdown(
                choices=["SavedModel (Actual en memoria)"],
                value="SavedModel (Actual en memoria)",
                label="📦 Formato del Modelo",
                info="Selecciona el formato específico (TFLite, SavedModel, etc.)"
            )

        gr.Markdown("#### 📸 Galería de Imágenes")
        gr.Markdown("Sube múltiples imágenes. La imagen seleccionada aparecerá con contorno azul.")

        # File upload for multiple images
        upload_images_btn = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="📤 Subir Imágenes (múltiples)"
        )

        # Gallery for multiple images with selection
        image_gallery = gr.Gallery(
            label="Imágenes para Inferencia - Haz clic para seleccionar",
            show_label=True,
            elem_id="inference_gallery",
            columns=[4],
            rows=[2],
            object_fit="cover",
            height=400,
            allow_preview=False
        )

        # Show selected image
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("##### 🎯 Imagen Seleccionada:")
                selected_image_display = gr.Image(
                    label="Imagen a Predecir",
                    type="pil",
                    interactive=False,
                    show_label=False
                )
            with gr.Column(scale=1):
                gr.Markdown("##### ℹ️ Información:")
                selection_info = gr.Textbox(
                    label="Estado",
                    value="Sube imágenes y selecciona una de la galería",
                    lines=6,
                    interactive=False
                )

        # Hidden state to store all images and selected index
        all_images_state = gr.State([])
        selected_image_index = gr.State(0)

        predict_btn = gr.Button(f"🔮 {t('predict_button')}", variant="primary")
        prediction_output = gr.Textbox(label=t('prediction_result'), lines=15)

        def refresh_prediction_models():
            models = get_available_trained_models()
            choices = [("Modelo Actual", None)] + models
            return gr.Dropdown(choices=choices)

        def update_format_choices(model_selection):
            """Update available formats when model selection changes"""
            formats = get_available_formats_for_model(model_selection)
            return gr.Dropdown(choices=formats, value=formats[0] if formats else None)

        def resize_image_for_gallery(img, size=300):
            """Resize image to uniform size for gallery display"""
            # Calculate aspect ratio
            img_ratio = img.width / img.height

            # Create square thumbnail
            if img_ratio > 1:
                # Landscape
                new_width = size
                new_height = int(size / img_ratio)
            else:
                # Portrait
                new_height = size
                new_width = int(size * img_ratio)

            # Resize and create square canvas
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create square background
            square_img = Image.new('RGB', (size, size), (240, 240, 240))

            # Paste resized image in center
            offset_x = (size - new_width) // 2
            offset_y = (size - new_height) // 2
            square_img.paste(resized, (offset_x, offset_y))

            return square_img

        def upload_multiple_images(files):
            """Process uploaded files and return list of images for gallery"""
            if files is None or len(files) == 0:
                return [], [], None, "❌ No se subieron imágenes", 0

            original_images = []
            gallery_images = []

            for file in files:
                try:
                    # Load original image
                    img = Image.open(file.name)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    original_images.append(img.copy())

                    # Create uniform sized version for gallery
                    gallery_img = resize_image_for_gallery(img.copy(), size=300)
                    gallery_images.append(gallery_img)

                except Exception as e:
                    print(f"Error loading image {file.name}: {e}")

            if len(original_images) == 0:
                return [], [], None, "❌ No se pudieron cargar las imágenes", 0

            # Select first image by default
            first_image = original_images[0]
            info_text = f"📊 Total: {len(original_images)} imágenes\n🎯 Seleccionada: Imagen #1\n✅ Lista para predecir"

            return gallery_images, original_images, first_image, info_text, 0

        def update_selected_image(evt: gr.SelectData, original_images):
            """Update selected image when user clicks on gallery"""
            if not original_images or evt.index >= len(original_images):
                return None, "❌ Error al seleccionar imagen", evt.index

            selected_idx = evt.index
            selected_image = original_images[selected_idx]

            info_text = f"📊 Total: {len(original_images)} imágenes\n🎯 Seleccionada: Imagen #{selected_idx + 1}\n✅ Lista para predecir"

            return selected_image, info_text, selected_idx

        def predict_with_gallery(all_images, selected_idx, model_selection, model_format):
            """Perform inference on the selected image from gallery"""
            if not all_images or len(all_images) == 0:
                return "❌ No hay imágenes cargadas. Sube al menos una imagen primero."

            if selected_idx is None or selected_idx < 0 or selected_idx >= len(all_images):
                selected_idx = 0

            # Get the selected image (use original, not gallery version)
            selected_image = all_images[selected_idx]

            # Perform inference
            result = perform_inference(selected_image, model_selection, model_format)

            # Add info about which image was processed
            result = f"🖼️ **Imagen procesada:** #{selected_idx + 1} de {len(all_images)}\n\n{result}"

            return result

        refresh_prediction_btn.click(
            refresh_prediction_models,
            outputs=prediction_model_dropdown
        )

        prediction_model_dropdown.change(
            update_format_choices,
            inputs=[prediction_model_dropdown],
            outputs=[prediction_format_dropdown]
        )

        upload_images_btn.upload(
            upload_multiple_images,
            inputs=[upload_images_btn],
            outputs=[image_gallery, all_images_state, selected_image_display, selection_info, selected_image_index]
        )

        image_gallery.select(
            update_selected_image,
            inputs=[all_images_state],
            outputs=[selected_image_display, selection_info, selected_image_index]
        )

        predict_btn.click(
            predict_with_gallery,
            inputs=[all_images_state, selected_image_index, prediction_model_dropdown, prediction_format_dropdown],
            outputs=[prediction_output]
        )
    
    gr.Markdown("---")
    gr.Markdown(f"## 🧠 Rachael Platform")
    gr.Markdown(f"### 🚀 Soluciones Profesionales de IA")
    gpu_status = t('gpu_enabled') if tf.config.list_physical_devices('GPU') else t('cpu_mode')
    gr.Markdown(f"{t('gpu_status')} {gpu_status}")
    gr.Markdown(f"**🎯 Características:** {t('features_list') if 'features_list' in TRANSLATIONS[current_lang] else 'Transfer Learning • Data Augmentation • TFLite Quantization • Métricas Avanzadas'}")
    gr.Markdown(f"Más información: [rachael.vision](https://rachael.vision) | {t('version_info')}")
    gr.Markdown(t('copyright_text'))

if __name__ == "__main__":
    app.queue()  # Enable queuing for progress tracking
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)