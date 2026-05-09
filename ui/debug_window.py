
"""
debug_window.py
---------------
Modern debug window with enhanced visualization and metrics.
"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

from core.preprocessor import Preprocessor


class DebugWindow:
    def __init__(self, parent, detection_results: list, colors: dict = None):
        self.preprocessor = Preprocessor()
        
        # Default colors if not provided
        self.colors = colors or {
            'primary': '#1a73e8',
            'success': '#0f9d58',
            'danger': '#ea4335',
            'warning': '#f4b400',
            'bg_main': '#f8f9fa',
            'bg_card': '#ffffff',
            'text_primary': '#202124',
            'text_secondary': '#5f6368',
            'border': '#dadce0',
        }
        
        self.win = tk.Toplevel(parent)
        self.win.title("🔬 Plate Debug Information")
        self.win.geometry("1200x800")
        self.win.configure(bg=self.colors['bg_main'])
        
        # Make window resizable
        self.win.minsize(900, 600)
        
        self._build_ui(detection_results)

    def _build_ui(self, detection_results: list):
        """Build the debug UI."""
        # Header
        header = tk.Frame(self.win, bg=self.colors['bg_main'])
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(header, text="🔬", font=('Segoe UI', 24),
                bg=self.colors['bg_main']).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(header, text="Plate Debug Information",
                font=('Segoe UI', 18, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_main']).pack(side=tk.LEFT)
        
        tk.Label(header, text=f"{len(detection_results)} plate(s) detected",
                font=('Segoe UI', 10),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_main']).pack(side=tk.RIGHT)
        
        # Notebook for multiple plates
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Style the notebook
        style = ttk.Style()
        style.configure('TNotebook', background=self.colors['bg_main'])
        style.configure('TNotebook.Tab', 
                       font=('Segoe UI', 10, 'bold'),
                       padding=[20, 8])
        
        for i, result in enumerate(detection_results):
            frame = tk.Frame(notebook, bg=self.colors['bg_main'])
            notebook.add(frame, text=f"  Plate {i+1}  ")
            self._build_tab(frame, result)

    def _build_tab(self, frame: tk.Frame, result: dict):
        """Build individual plate tab."""
        try:
            plate_img = result.get('plate_img')
            
            # Main content area with two columns
            content = tk.Frame(frame, bg=self.colors['bg_main'])
            content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Left column - Images
            left_col = tk.Frame(content, bg=self.colors['bg_main'])
            left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            # Right column - Info
            right_col = tk.Frame(content, bg=self.colors['bg_main'])
            right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
            
            if plate_img is not None:
                # Original image
                self._build_image_card(left_col, plate_img, 
                                      "📷 Original Detection",
                                      is_color=True)
                
                # Processed image
                processed = self.preprocessor.process(plate_img)
                self._build_image_card(left_col, processed,
                                      "🔧 After Preprocessing",
                                      is_color=False)
                
                # Processing steps visualization
                self._build_processing_steps(left_col, plate_img)
            
            # OCR Results
            self._build_ocr_results(right_col, result)
            
            # Image metrics
            if plate_img is not None:
                self._build_metrics(right_col, plate_img)
            
            # Confidence visualization
            self._build_confidence_meter(right_col, result)
            
        except Exception as e:
            tk.Label(frame, text=f"Error: {e}",
                    font=('Segoe UI', 11),
                    fg=self.colors['danger'],
                    bg=self.colors['bg_main']).pack(pady=20)

    def _build_image_card(self, parent, img, title: str, is_color: bool = True):
        """Build an image display card."""
        # Card frame
        card = tk.Frame(parent, bg=self.colors['bg_card'],
                       relief='flat', bd=0, highlightthickness=1,
                       highlightbackground=self.colors['border'])
        card.pack(fill=tk.X, pady=(0, 10))
        
        # Card header
        card_header = tk.Frame(card, bg=self.colors['bg_card'])
        card_header.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        tk.Label(card_header, text=title,
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(anchor=tk.W)
        
        # Image display
        if is_color and len(img.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(img)
        
        # Resize maintaining aspect ratio
        display_width = 400
        w_percent = display_width / float(pil_img.size[0])
        h_size = int(float(pil_img.size[1]) * float(w_percent))
        pil_img = pil_img.resize((display_width, h_size), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_img)
        
        img_label = tk.Label(card, image=photo, bg=self.colors['bg_card'])
        img_label.image = photo
        img_label.pack(pady=(5, 10))
        
        # Size info
        h, w = img.shape[:2] if len(img.shape) > 2 else img.shape
        tk.Label(card, text=f"Size: {w}×{h} px",
                font=('Segoe UI', 8),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_card']).pack(pady=(0, 10))

    def _build_processing_steps(self, parent, plate_img):
        """Show processing steps visualization."""
        card = tk.Frame(parent, bg=self.colors['bg_card'],
                       relief='flat', bd=0, highlightthickness=1,
                       highlightbackground=self.colors['border'])
        card.pack(fill=tk.X)
        
        tk.Label(card, text="🔄 Processing Pipeline",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(padx=15, pady=(10, 5))
        
        steps = [
            ("1. Grayscale", "Convert to grayscale"),
            ("2. Upscale", "Increase resolution"),
            ("3. Denoise", "Remove noise"),
            ("4. Enhance", "Improve contrast"),
            ("5. Threshold", "Binarize image"),
            ("6. Clean", "Morphological ops"),
        ]
        
        for step_name, step_desc in steps:
            step_frame = tk.Frame(card, bg=self.colors['bg_card'])
            step_frame.pack(fill=tk.X, padx=15, pady=3)
            
            tk.Label(step_frame, text=step_name,
                    font=('Segoe UI', 9, 'bold'),
                    fg=self.colors['primary'],
                    bg=self.colors['bg_card'],
                    width=15, anchor=tk.W).pack(side=tk.LEFT)
            
            tk.Label(step_frame, text=step_desc,
                    font=('Segoe UI', 9),
                    fg=self.colors['text_secondary'],
                    bg=self.colors['bg_card']).pack(side=tk.LEFT)
            
            # Checkmark
            tk.Label(step_frame, text="✓",
                    font=('Segoe UI', 9),
                    fg=self.colors['success'],
                    bg=self.colors['bg_card']).pack(side=tk.RIGHT, padx=5)

    def _build_ocr_results(self, parent, result: dict):
        """Build OCR results display."""
        card = tk.Frame(parent, bg=self.colors['bg_card'],
                       relief='flat', bd=0, highlightthickness=1,
                       highlightbackground=self.colors['border'])
        card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(card, text="📝 OCR Results",
                font=('Segoe UI', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(padx=20, pady=(15, 10))
        
        # Plate text with large font
        plate_text = result.get('plate_text', 'N/A')
        text_frame = tk.Frame(card, bg='#f0f0f0', relief='flat')
        text_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Label(text_frame, text=plate_text,
                font=('Courier New', 24, 'bold'),
                fg=self.colors['text_primary'],
                bg='#f0f0f0').pack(padx=20, pady=15)
        
        # Status with colored badge
        is_blacklisted = result.get('is_blacklisted', False)
        reason = result.get('reason', '')
        
        info_frame = tk.Frame(card, bg=self.colors['bg_card'])
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Status badge
        if is_blacklisted:
            status_bg = '#fce8e6'
            status_fg = '#c5221f'
            status_text = '⚠ BLACKLISTED'
        else:
            status_bg = '#e6f4ea'
            status_fg = '#137333'
            status_text = '✓ CLEAN'
        
        badge = tk.Frame(info_frame, bg=status_bg)
        badge.pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(badge, text=f" {status_text} ",
                font=('Segoe UI', 10, 'bold'),
                fg=status_fg, bg=status_bg,
                padx=10, pady=3).pack()
        
        # Reason
        if reason:
            tk.Label(info_frame, text=reason,
                    font=('Segoe UI', 10),
                    fg=self.colors['text_secondary'],
                    bg=self.colors['bg_card']).pack(side=tk.LEFT)

    def _build_metrics(self, parent, plate_img):
        """Build image quality metrics."""
        card = tk.Frame(parent, bg=self.colors['bg_card'],
                       relief='flat', bd=0, highlightthickness=1,
                       highlightbackground=self.colors['border'])
        card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(card, text="📊 Image Quality Metrics",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(padx=20, pady=(15, 10))
        
        # Calculate metrics
        h, w = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        metrics = [
            ("Resolution", f"{w}×{h} px", "📐"),
            ("Sharpness", f"{blur_score:.1f}", "🔍"),
            ("Brightness", f"{brightness:.1f}", "💡"),
            ("Contrast", f"{contrast:.1f}", "🎨"),
        ]
        
        for metric_name, metric_value, icon in metrics:
            metric_frame = tk.Frame(card, bg=self.colors['bg_card'])
            metric_frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(metric_frame, text=f"{icon} {metric_name}:",
                    font=('Segoe UI', 10),
                    fg=self.colors['text_secondary'],
                    bg=self.colors['bg_card'],
                    width=15, anchor=tk.W).pack(side=tk.LEFT)
            
            tk.Label(metric_frame, text=metric_value,
                    font=('Segoe UI', 10, 'bold'),
                    fg=self.colors['text_primary'],
                    bg=self.colors['bg_card']).pack(side=tk.LEFT)
        
        # Quality indicator
        if blur_score > 200:
            quality = "Excellent"
            quality_color = self.colors['success']
        elif blur_score > 100:
            quality = "Good"
            quality_color = self.colors['primary']
        elif blur_score > 50:
            quality = "Fair"
            quality_color = self.colors['warning']
        else:
            quality = "Poor"
            quality_color = self.colors['danger']
        
        quality_frame = tk.Frame(card, bg=self.colors['bg_card'])
        quality_frame.pack(fill=tk.X, padx=20, pady=(10, 15))
        
        tk.Label(quality_frame, text="Overall Quality:",
                font=('Segoe UI', 10),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_card']).pack(side=tk.LEFT)
        
        tk.Label(quality_frame, text=f" {quality} ",
                font=('Segoe UI', 10, 'bold'),
                fg=quality_color,
                bg=self.colors['bg_card']).pack(side=tk.LEFT, padx=5)

    def _build_confidence_meter(self, parent, result: dict):
        """Build confidence visualization."""
        card = tk.Frame(parent, bg=self.colors['bg_card'],
                       relief='flat', bd=0, highlightthickness=1,
                       highlightbackground=self.colors['border'])
        card.pack(fill=tk.X)
        
        tk.Label(card, text="🎯 Confidence Score",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(padx=20, pady=(15, 10))
        
        # Calculate confidence
        plate_text = result.get('plate_text', '')
        is_blacklisted = result.get('is_blacklisted', False)
        
        if plate_text in ("UNREADABLE", "ERROR"):
            confidence = 0
        elif is_blacklisted:
            confidence = 95
        else:
            confidence = 85  # Default for clean reads
        
        # Progress bar
        bar_frame = tk.Frame(card, bg='#f0f0f0', height=30)
        bar_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        bar_frame.pack_propagate(False)
        
        # Filled portion
        if confidence > 0:
            if confidence >= 80:
                fill_color = self.colors['success']
            elif confidence >= 60:
                fill_color = self.colors['warning']
            else:
                fill_color = self.colors['danger']
            
            fill_width = int((confidence / 100) * 360)  # Approximate max width
            fill = tk.Frame(bar_frame, bg=fill_color, width=fill_width, height=30)
            fill.place(x=0, y=0)
            fill.pack_propagate(False)
        
        tk.Label(bar_frame, text=f"{confidence}%",
                font=('Segoe UI', 10, 'bold'),
                fg='white' if confidence > 50 else self.colors['text_primary'],
                bg=fill_color if confidence > 0 else '#f0f0f0').place(relx=0.5, rely=0.5, anchor=tk.CENTER)