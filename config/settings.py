import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "50w_best.pt")
BLACKLIST_PATH  = os.path.join(BASE_DIR, "data",   "blacklist.csv")

# ── YOLO Detection ─────────────────────────────────────────────────────
YOLO_CONFIDENCE     = 0.3
PLATE_PADDING       = 5
MIN_PLATE_ASPECT    = 1.5
MAX_PLATE_ASPECT    = 6.0
MAX_PLATE_WIDTH     = 600
MAX_PLATE_HEIGHT    = 200

# ── Contour-based plate finder ─────────────────────────────────────────
CONTOUR_MIN_ASPECT  = 2.0
CONTOUR_MAX_ASPECT  = 6.0
CONTOUR_MIN_AREA    = 1500
CONTOUR_MIN_W       = 80
CONTOUR_MIN_H       = 20
CONTOUR_MAX_H       = 80
SOBEL_DENSITY_MIN   = 0.1

# ── Preprocessing ──────────────────────────────────────────────────────
TARGET_OCR_WIDTH    = 400
TARGET_OCR_HEIGHT   = 120
CROP_TOP_RATIO      = 0.10
CROP_BOTTOM_RATIO   = 0.90
DARK_BG_THRESHOLD   = 0.60

# ── Enhanced Preprocessing Settings ────────────────────────────────────
MIN_PLATE_HEIGHT_FOR_UPSCALE = 50    # Height threshold for aggressive upscaling
MIN_PLATE_WIDTH_FOR_UPSCALE  = 150   # Width threshold for aggressive upscaling
MIN_UPSCALE_FACTOR           = 4.0   # Minimum upscale factor for tiny plates
TARGET_MIN_HEIGHT            = 200.0 # Target minimum height after upscaling
MAX_UPSCALE_FACTOR           = 8.0   # Maximum upscale factor cap
DEFAULT_UPSCALE_FACTOR       = 2.5   # Default upscale factor for normal plates

# CLAHE Parameters
CLAHE_CLIP_LIMITS = [2.0, 3.0, 4.0]     # Different CLAHE clip limits to try
CLAHE_TILE_SIZES  = [(8,8), (4,4), (12,12)]  # Different tile sizes

# Denoising Parameters
DENOISE_STRENGTH    = 10    # h parameter for non-local means denoising
DENOISE_TEMPLATE    = 7     # templateWindowSize
DENOISE_SEARCH      = 21    # searchWindowSize

# Gamma Correction
GAMMA_VALUE         = 1.5   # Gamma correction value for dark plates

# Sharpening Kernels
SHARPEN_KERNEL_STRONG = [[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]

SHARPEN_KERNEL_MILD   = [[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]]

# Edge Detection for Deskew
CANNY_LOW_THRESHOLD  = 50
CANNY_HIGH_THRESHOLD = 150
CANNY_APERTURE       = 5
HOUGH_LINES_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP    = 5
DESKEW_ANGLE_MIN      = -10    # Minimum angle to consider for deskew
DESKEW_ANGLE_MAX      = 10     # Maximum angle to consider
DESKEW_ANGLE_SKIP     = 0.3    # Skip if angle is less than this

# Adaptive Threshold Parameters
ADAPTIVE_THRESH_BLOCK_SIZES = [15, 11]
ADAPTIVE_THRESH_C_VALUES    = [6, 4]

# Morphological Cleaning
MORPH_KERNEL_SIZE    = (2, 2)
MIN_COMPONENT_AREA   = 10     # Minimum area to keep after cleaning

# Sauvola Threshold (if available)
SAUVOLA_WINDOW_FACTOR = 4    # Window size = min(width, height) / factor

# ── OCR ────────────────────────────────────────────────────────────────
TESSERACT_CONF_MIN  = 30       # Lower threshold to get more candidates
EASYOCR_CONF_MIN    = 0.4      # Lower threshold for EasyOCR
EASYOCR_MIN_SIZE    = 10       # Minimum text size for EasyOCR

# Tesseract Configurations
TESSERACT_CONFIGS = [
    '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
]

# EasyOCR Parameters
EASYOCR_WIDTH_THS   = 0.5      # Width threshold for text grouping
EASYOCR_HEIGHT_THS  = 0.5      # Height threshold for text grouping
EASYOCR_MIN_CONF_SINGLE = 0.4  # Minimum confidence for single detections

# Tesseract Word Confidence
TESSERACT_MIN_WORD_CONF = 30   # Minimum confidence for individual words

# Characters to strip globally (separators)
OCR_GLOBAL_REMOVE = {' ': '', '.': '', '-': ''}

# Letter→digit corrections applied only at digit positions
OCR_DIGIT_CORRECTIONS = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'L': '1',
    'D': '0', 'G': '6', 'T': '7', 'B': '8', 'A': '4',
    'Q': '0', 'U': '0', 'E': '3', 'F': '7',
}

# Letter corrections for state code positions (0 and 1)
OCR_LETTER_CORRECTIONS = {
    '1': 'I', '0': 'O', '2': 'Z', '5': 'S', '8': 'B',
    '4': 'A', '6': 'G', '3': 'E', '7': 'T',
}

# Common single-letter confusions at position 0-1
OCR_STATE_CODE_FIXES = {
    'IK': 'JK', '1K': 'JK', 'IH': 'JH', '1H': 'JH',
    'WI': 'MH', 'IVI': 'MH', 'IW': 'MH',
}

# ── Valid Indian State Codes ───────────────────────────────────────────
INDIAN_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LD', 'MH', 'ML',
    'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN',
    'TR', 'TS', 'UK', 'UP', 'WB', 'AN', 'LA', 'DN'
}

# ── Indian Plate Format Patterns ───────────────────────────────────────
INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # XX00XX0000
    r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',        # XX00X0000
    r'^[A-Z]{2}\d{2}[A-Z]{2}\d{3}$',     # XX00XX000
    r'^[A-Z]{2}\d{2}[A-Z]\d{3}$',        # XX00X000
    r'^[A-Z]{2}\d{2}\d{4}$',             # XX000000
]

# ── Prefixes to Remove ─────────────────────────────────────────────────
PLATE_PREFIXES_TO_REMOVE = ['IND', 'ND']

# ── Plate Similarity Threshold ─────────────────────────────────────────
PLATE_SIMILARITY_THRESHOLD = 0.8  # For grouping similar plate readings

# ── Position-Aware Correction Rules ────────────────────────────────────
# Position 0-1: State code position digits that should be letters
STATE_CODE_DIGIT_TO_LETTER = {
    '1': 'I',
    '0': 'O',
}

# Position 2-3: District code position letters that should be digits
DISTRICT_CODE_LETTER_TO_DIGIT = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'S': '5',
}

# Position last-4: Number position letters that should be digits
NUMBER_LETTER_TO_DIGIT = {
    'O': '0',
    'I': '1',
    'S': '5',
}

# ── Confidence Scoring Weights ─────────────────────────────────────────
VALID_PLATE_BONUS        = 15    # Bonus confidence for valid plate format
VALID_STATE_CODE_BONUS   = 10    # Bonus for valid state code
PARTIAL_PLATE_MULTIPLIER = 0.7   # Multiplier for partial matches
OCR_ENGINE_BONUS         = 5     # Bonus for matching results from multiple engines

# ── Minimum Plate Requirements ─────────────────────────────────────────
MIN_PLATE_TEXT_LENGTH     = 4     # Minimum characters for a valid plate
MIN_PLATE_LENGTH_PARTIAL  = 5     # Minimum length for partial matches
MIN_PLATE_LENGTH_VALID    = 6     # Minimum length for valid plate
MAX_PLATE_LENGTH          = 10    # Maximum valid plate length

# ── Blacklist ──────────────────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD = 0.85

# ── UI ─────────────────────────────────────────────────────────────────
WINDOW_SIZE     = "1200x800"
DEBUG_WIN_SIZE  = "1000x700"
DISPLAY_MAX_W   = 800
DISPLAY_MAX_H   = 600