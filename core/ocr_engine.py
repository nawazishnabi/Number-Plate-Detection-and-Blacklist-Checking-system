"""
ocr_engine.py
-------------
OCR layer optimized for Indian license plates.
Uses multiple engines and smart corrections.
All configuration values are imported from settings.py
"""

import re
import cv2
import numpy as np
import pytesseract
import easyocr
from collections import Counter
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TESSERACT_CONFIGS, TESSERACT_CONF_MIN, EASYOCR_CONF_MIN,
    OCR_GLOBAL_REMOVE, OCR_DIGIT_CORRECTIONS, OCR_LETTER_CORRECTIONS,
    OCR_STATE_CODE_FIXES, INDIAN_STATE_CODES, INDIAN_PLATE_PATTERNS,
    PLATE_PREFIXES_TO_REMOVE, PLATE_SIMILARITY_THRESHOLD,
    STATE_CODE_DIGIT_TO_LETTER, DISTRICT_CODE_LETTER_TO_DIGIT,
    NUMBER_LETTER_TO_DIGIT, VALID_PLATE_BONUS, PARTIAL_PLATE_MULTIPLIER,
    OCR_ENGINE_BONUS, MIN_PLATE_TEXT_LENGTH, MIN_PLATE_LENGTH_PARTIAL,
    MIN_PLATE_LENGTH_VALID, MAX_PLATE_LENGTH, EASYOCR_WIDTH_THS,
    EASYOCR_HEIGHT_THS, EASYOCR_MIN_SIZE, TESSERACT_MIN_WORD_CONF,
    EASYOCR_MIN_CONF_SINGLE,
)
from core.preprocessor import Preprocessor


class OCREngine:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("[OCR] EasyOCR reader initialised")

    def extract_text(self, plate_img: np.ndarray) -> str:
        """Extract and correct plate text using ensemble approach."""
        try:
            # Step 1: Get all preprocessed variants
            variants = self.preprocessor.process_all(plate_img)
            
            # Step 2: Collect ALL candidates from ALL methods
            all_candidates = []
            
            # Run Tesseract on all variants
            for i, variant in enumerate(variants):
                tesseract_results = self._run_tesseract(variant, f"variant_{i}")
                all_candidates.extend(tesseract_results)
            
            # Run EasyOCR on original and best preprocessed variants
            easyocr_results = self._run_easyocr(plate_img, "original")
            all_candidates.extend(easyocr_results)
            
            # Only run EasyOCR on top 2 preprocessed variants
            for i, variant in enumerate(variants[:2]):
                easyocr_results = self._run_easyocr(variant, f"variant_{i}")
                all_candidates.extend(easyocr_results)
            
            if not all_candidates:
                return "UNREADABLE"
            
            # Step 3: Apply smart voting
            best_text = self._ensemble_vote(all_candidates)
            
            if best_text:
                print(f"[OCR] Selected: '{best_text}' from {len(all_candidates)} candidates")
                return best_text
            else:
                return "UNREADABLE"
                
        except Exception as e:
            print(f"[OCR] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR"

    def _run_tesseract(self, img: np.ndarray, label: str) -> list:
        """Run Tesseract with multiple configurations."""
        results = []
        
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        for config in TESSERACT_CONFIGS:
            try:
                # Get text
                raw_text = pytesseract.image_to_string(img, config=config)
                cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                
                if len(cleaned) < MIN_PLATE_TEXT_LENGTH:
                    continue
                
                # Get confidence scores
                data = pytesseract.image_to_data(
                    img, config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [
                    int(c) for c in data['conf'] 
                    if str(c).lstrip('-').isdigit() and int(c) > 0
                ]
                
                avg_conf = sum(confidences) / len(confidences) if confidences else 50
                
                # Basic correction for obvious errors
                corrected = self._basic_correction(cleaned)
                
                if self._is_valid_indian_plate(corrected):
                    score = avg_conf + VALID_PLATE_BONUS
                    results.append((corrected, score, f"Tesseract/{label}"))
                elif len(corrected) >= MIN_PLATE_LENGTH_PARTIAL:
                    score = avg_conf * PARTIAL_PLATE_MULTIPLIER
                    results.append((corrected, score, f"Tesseract/{label}/partial"))
                    
            except Exception as e:
                print(f"[OCR] Tesseract error: {e}")
        
        return results

    def _run_easyocr(self, img: np.ndarray, label: str) -> list:
        """Run EasyOCR with proper preprocessing."""
        results = []
        
        try:
            # Convert to RGB if needed
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    ocr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    ocr_img = img
            else:
                ocr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Run EasyOCR with settings from config
            detections = self.reader.readtext(
                ocr_img,
                detail=1,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                width_ths=EASYOCR_WIDTH_THS,
                height_ths=EASYOCR_HEIGHT_THS,
                min_size=EASYOCR_MIN_SIZE,
            )
            
            if detections:
                # Sort by x-coordinate (left to right)
                sorted_dets = sorted(detections, key=lambda d: d[0][0][0])
                
                # Build merged text
                merged_text = ''.join(d[1] for d in sorted_dets)
                merged_conf = sum(d[2] for d in sorted_dets) / len(sorted_dets)
                
                # Clean the text
                cleaned = re.sub(r'[^A-Z0-9]', '', merged_text.upper())
                
                # Remove prefixes specified in settings
                for prefix in PLATE_PREFIXES_TO_REMOVE:
                    cleaned = re.sub(f'^{prefix}', '', cleaned)
                
                # Basic correction
                corrected = self._basic_correction(cleaned)
                
                if len(corrected) >= MIN_PLATE_TEXT_LENGTH:
                    score = merged_conf * 100
                    if self._is_valid_indian_plate(corrected):
                        score += VALID_PLATE_BONUS
                    results.append((corrected, score, f"EasyOCR/{label}"))
                
                # Also add individual detections as backup
                for _, text, conf in sorted_dets:
                    if conf > EASYOCR_MIN_CONF_SINGLE:
                        cleaned_item = re.sub(r'[^A-Z0-9]', '', text.upper())
                        for prefix in PLATE_PREFIXES_TO_REMOVE:
                            cleaned_item = re.sub(f'^{prefix}', '', cleaned_item)
                        
                        corrected_item = self._basic_correction(cleaned_item)
                        
                        if self._is_valid_indian_plate(corrected_item):
                            results.append((corrected_item, conf * 100 + OCR_ENGINE_BONUS, 
                                          f"EasyOCR-single/{label}"))
        
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
        
        return results

    def _basic_correction(self, text: str) -> str:
        """
        Apply ONLY basic, high-confidence corrections using settings.
        DON'T try to fix things that might be correct.
        """
        if len(text) < MIN_PLATE_TEXT_LENGTH:
            return text
        
        # Remove common prefixes from settings
        for prefix in PLATE_PREFIXES_TO_REMOVE:
            text = re.sub(f'^{prefix}', '', text)
        
        chars = list(text)
        
        # Position 0-1: State code (should be letters)
        for i in range(min(2, len(chars))):
            if chars[i] in STATE_CODE_DIGIT_TO_LETTER:
                chars[i] = STATE_CODE_DIGIT_TO_LETTER[chars[i]]
        
        # Position 2-3: District code (should be digits)
        for i in range(2, min(4, len(chars))):
            if chars[i] in DISTRICT_CODE_LETTER_TO_DIGIT:
                chars[i] = DISTRICT_CODE_LETTER_TO_DIGIT[chars[i]]
        
        # Last 4 positions: Should be digits
        if len(chars) >= MIN_PLATE_LENGTH_VALID + 2:
            for i in range(len(chars) - 4, len(chars)):
                if chars[i] in NUMBER_LETTER_TO_DIGIT:
                    chars[i] = NUMBER_LETTER_TO_DIGIT[chars[i]]
        
        return ''.join(chars)

    def _is_valid_indian_plate(self, text: str) -> bool:
        """Check if text matches Indian plate format using settings."""
        if len(text) < MIN_PLATE_LENGTH_VALID or len(text) > MAX_PLATE_LENGTH:
            return False
        
        # Check state code from settings
        state_code = text[:2]
        if state_code not in INDIAN_STATE_CODES:
            return False
        
        # Check format patterns from settings
        for pattern in INDIAN_PLATE_PATTERNS:
            if re.match(pattern, text):
                return True
        
        return False

    def _ensemble_vote(self, candidates: list) -> str:
        """
        Smart voting: prefer candidates with valid state codes and format.
        """
        if not candidates:
            return ""
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Log top candidates
        print(f"[OCR] Top candidates:")
        for text, conf, method in candidates[:5]:
            print(f"  '{text}' (conf: {conf:.1f}, via: {method})")
        
        # Separate valid and invalid candidates
        valid_plates = []
        invalid_plates = []
        
        for text, conf, method in candidates:
            if self._is_valid_indian_plate(text):
                valid_plates.append((text, conf, method))
            else:
                invalid_plates.append((text, conf, method))
        
        # If we have valid plates, use frequency-based selection
        if valid_plates:
            # Count occurrences of each plate (similar plates count as same)
            plate_groups = {}
            for text, conf, method in valid_plates:
                # Find if this belongs to an existing group
                found_group = None
                for group_key in plate_groups:
                    if self._plates_similar(text, group_key):
                        found_group = group_key
                        break
                
                if found_group:
                    plate_groups[found_group].append((text, conf))
                else:
                    plate_groups[text] = [(text, conf)]
            
            # Select the group with most votes
            best_group_key = max(plate_groups, key=lambda k: len(plate_groups[k]))
            
            # Within best group, select highest confidence
            best_candidate = max(plate_groups[best_group_key], key=lambda x: x[1])
            
            return best_candidate[0]
        
        # Fallback: use best invalid candidate if nothing valid
        elif invalid_plates:
            # Try to fix the best invalid candidate
            best_text = invalid_plates[0][0]
            fixed_text = self._try_fix_state_code(best_text)
            
            if fixed_text != best_text and self._is_valid_indian_plate(fixed_text):
                return fixed_text
            
            return best_text
        
        return ""

    def _plates_similar(self, plate1: str, plate2: str) -> bool:
        """Check if two plate readings are likely the same using settings threshold."""
        if abs(len(plate1) - len(plate2)) > 1:
            return False
        
        # Check if they share the same pattern
        min_len = min(len(plate1), len(plate2))
        matches = sum(1 for i in range(min_len) if plate1[i] == plate2[i])
        similarity = matches / min_len
        
        return similarity >= PLATE_SIMILARITY_THRESHOLD

    def _try_fix_state_code(self, text: str) -> str:
        """Try to fix invalid state code using settings."""
        if len(text) < MIN_PLATE_LENGTH_VALID:
            return text
        
        state_code = text[:2]
        
        # If current state code is already valid, DON'T CHANGE IT
        if state_code in INDIAN_STATE_CODES:
            return text
        
        # Check if one char off from a valid code
        for valid_code in INDIAN_STATE_CODES:
            if self._levenshtein_distance(state_code, valid_code) == 1:
                # Only change if we're very confident (first or second char matches)
                if state_code[0] == valid_code[0] or state_code[1] == valid_code[1]:
                    return valid_code + text[2:]
        
        return text

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]