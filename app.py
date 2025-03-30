from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
print("Loading ResNet50 model...")
model = ResNet50(weights='imagenet')
print("ResNet50 model loaded successfully")

# Load the saved model if it exists
try:
    print("Loading custom model...")
    loaded_model = tf.keras.models.load_model('saved_model.keras')
    print("Custom model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load custom model: {str(e)}")
    loaded_model = None

def is_valid_image(file_stream):
    try:
        image = Image.open(file_stream)
        image.verify()
        return True
    except Exception as e:
        print(f"Image validation error: {str(e)}")
        return False

def predict_image(img_path):
    try:
        # Open and validate image
        with Image.open(img_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize((224, 224))
            
            # Convert to array
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Make prediction
            predictions = model.predict(x)
            results = decode_predictions(predictions, top=10)[0]
            
            # Define broader set of animal terms
            animal_terms = {
                'dog': ['golden_retriever', 'labrador', 'retriever', 'german_shepherd', 'husky', 
                       'beagle', 'poodle', 'bulldog', 'rottweiler', 'collie', 'terrier', 'spaniel',
                       'puppy', 'hound', 'hunting_dog', 'sporting_dog'],
                'cat': ['tabby', 'persian_cat', 'siamese_cat', 'maine_coon', 'egyptian_cat'],
                'other_animals': ['hedgehog', 'hamster', 'rabbit', 'guinea_pig', 'mouse', 'rat',
                                'bird', 'parrot', 'canary', 'turtle', 'fish', 'lizard', 'snake',
                                'iguana', 'frog', 'toad', 'salamander', 'porcupine']
            }
            
            # Check predictions
            for pred in results:
                class_name = pred[1].lower()
                
                # Check for dogs
                if any(breed in class_name for breed in animal_terms['dog']):
                    return "Dog"
                # Check for cats
                elif any(breed in class_name for breed in animal_terms['cat']):
                    return "Cat"
                # Check for other animals
                elif any(animal in class_name for animal in animal_terms['other_animals']):
                    # Capitalize the matched animal name
                    for animal in animal_terms['other_animals']:
                        if animal in class_name:
                            return animal.title()
            
            # If no specific animal is found, return the top prediction's class name
            return results[0][1].replace('_', ' ').title()
                
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def predict_image_from_memory(img):
    try:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize((224, 224))
        
        # Convert to array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        predictions = model.predict(x)
        results = decode_predictions(predictions, top=10)[0]
        
        # Return the prediction
        return results[0][1].replace('_', ' ').title()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the file into memory
        file_content = file.read()
        
        # Create a BytesIO object
        image_stream = io.BytesIO(file_content)
        
        # Validate the image
        if not is_valid_image(image_stream):
            return jsonify({'error': 'Invalid image file. Please ensure you are uploading a valid image file (JPG, PNG, etc.)'}), 400
        
        # Reset the stream position
        image_stream.seek(0)
        
        # Use PIL to open the image from memory
        with Image.open(image_stream) as img:
            prediction = predict_image_from_memory(img)
            
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)