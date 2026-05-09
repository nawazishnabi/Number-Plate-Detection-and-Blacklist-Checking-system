"""
wsgi.py
-------
WSGI entry point for production deployment on Render.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Flask app
from web_app import app  

# Set Tesseract path for Linux environment
import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH', '/usr/bin/tesseract')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


# This must exist for the import to work
app = Flask(__name__)