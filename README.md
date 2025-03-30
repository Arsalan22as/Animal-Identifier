# Animal Image Classifier

A web application that can identify animals from uploaded images using deep learning. The application uses the pre-trained ResNet50 model to classify images and identify various animals.

## Features

- Drag and drop image upload
- Real-time image preview
- Animal classification with confidence scores
- Modern, responsive UI
- Support for common image formats

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the upload area or drag and drop an image file
2. Wait for the image to be processed
3. View the predictions with confidence scores

## Technical Details

- Backend: Flask (Python)
- Frontend: HTML, JavaScript, TailwindCSS
- Model: ResNet50 (pre-trained on ImageNet)
- Image Processing: TensorFlow/Keras

## Notes

- Maximum file size: 16MB
- Supported image formats: JPG, PNG, GIF, etc.
- The model is optimized for animal classification.