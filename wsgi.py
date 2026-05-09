import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app.app import app  # ← try this instead of "from web_app import app"

import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH', '/usr/bin/tesseract')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)