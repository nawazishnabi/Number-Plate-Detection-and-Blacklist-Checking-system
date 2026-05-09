"""
web_app/app.py
--------------
Flask application optimized for Render deployment.
Features: async processing, disk storage, error handling.
"""

import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
import time
import logging
from pathlib import Path

# Set Tesseract path for Linux
if os.environ.get('RENDER') or not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Import core modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.detector import PlateDetector
from core.ocr_engine import OCREngine
from core.blacklist import BlacklistManager

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('/tmp', 'uploads') if os.environ.get('RENDER') else 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year cache for static files

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Initialize components with error handling
try:
    detector = PlateDetector()
    logger.info("✅ Plate detector initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize detector: {e}")
    detector = None

try:
    ocr = OCREngine()
    logger.info("✅ OCR engine initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize OCR: {e}")
    ocr = None

try:
    blacklist = BlacklistManager()
    logger.info(f"✅ Blacklist loaded with {len(blacklist.df)} entries")
except Exception as e:
    logger.error(f"❌ Failed to load blacklist: {e}")
    blacklist = None


@app.route('/')
def index():
    """Render main application page."""
    return render_template('index.html', 
                         version="1.0.0",
                         blacklist_count=len(blacklist.df) if blacklist else 0)


@app.route('/api/health')
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'components': {
            'detector': detector is not None,
            'ocr': ocr is not None,
            'blacklist': blacklist is not None
        }
    })


@app.route('/api/detect', methods=['POST'])
def detect_plates():
    """API endpoint for plate detection."""
    start_time = time.time()
    
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}
        if not '.' in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        
        # Check components
        if not detector:
            return jsonify({'error': 'Detector not initialized'}), 500
        
        # Detect plates
        plate_images = detector.detect_plates(filepath)
        
        if not plate_images:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': True,
                'plates': [],
                'message': 'No plates detected in the image',
                'processing_time': round(time.time() - start_time, 2)
            })
        
        # Process each plate
        results = []
        for i, plate_img in enumerate(plate_images):
            try:
                # Extract text
                text = ocr.extract_text(plate_img) if ocr else "ERROR"
                
                # Check blacklist
                if blacklist:
                    is_blacklisted, reason = blacklist.check(text)
                else:
                    is_blacklisted, reason = False, ""
                
                # Determine status
                if text in ('UNREADABLE', 'ERROR'):
                    status = 'unreadable'
                    if not reason:
                        reason = 'Could not extract text clearly'
                elif is_blacklisted:
                    status = 'blacklisted'
                else:
                    status = 'clean'
                    reason = 'Vehicle is not in blacklist'
                
                # Convert plate image to base64
                try:
                    _, buffer = cv2.imencode('.jpg', plate_img, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                    plate_base64 = base64.b64encode(buffer).decode('utf-8')
                    plate_data_url = f'data:image/jpeg;base64,{plate_base64}'
                except:
                    plate_data_url = None
                
                results.append({
                    'index': i + 1,
                    'plate_text': text,
                    'status': status,
                    'reason': reason,
                    'is_blacklisted': is_blacklisted,
                    'plate_image': plate_data_url,
                    'plate_size': {
                        'width': plate_img.shape[1],
                        'height': plate_img.shape[0]
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing plate {i}: {e}")
                results.append({
                    'index': i + 1,
                    'plate_text': 'ERROR',
                    'status': 'error',
                    'reason': str(e),
                    'is_blacklisted': False,
                    'plate_image': None
                })
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Processed {len(results)} plates in {processing_time}s")
        
        return jsonify({
            'success': True,
            'plates': results,
            'count': len(results),
            'processing_time': processing_time
        })
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'processing_time': round(time.time() - start_time, 2)
        }), 500


@app.route('/api/stats')
def get_stats():
    """Get system statistics."""
    return jsonify({
        'blacklist_entries': len(blacklist.df) if blacklist else 0,
        'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0,
        'model_loaded': detector is not None and detector.model is not None,
        'ocr_ready': ocr is not None,
    })


@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory('static', 'favicon.ico', 
                             mimetype='image/vnd.microsoft.icon')


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# Set start time
app.start_time = time.time()