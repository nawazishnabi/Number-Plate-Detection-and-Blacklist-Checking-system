
"""
app.py
------
Modern Tkinter application with professional UI/UX design.
Features: Dark theme, animated transitions, better layouts, real-time status.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
import threading

from core.detector   import PlateDetector
from core.ocr_engine import OCREngine
from core.blacklist  import BlacklistManager
from ui.debug_window import DebugWindow
from config.settings import WINDOW_SIZE, DISPLAY_MAX_W, DISPLAY_MAX_H


class LicensePlateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("License Plate Recognition System")
        self.root.geometry(WINDOW_SIZE)
        
        # Set minimum window size
        self.root.minsize(1000, 700)
        
        # Color scheme
        self.colors = {
            'primary':       '#1a73e8',  # Google Blue
            'primary_dark':  '#1557b0',
            'success':       '#0f9d58',  # Green
            'danger':        '#ea4335',  # Red
            'warning':       '#f4b400',  # Yellow/Orange
            'bg_main':       '#f8f9fa',  # Light gray
            'bg_card':       '#ffffff',  # White
            'text_primary':  '#202124',  # Dark gray
            'text_secondary':'#5f6368',  # Medium gray
            'border':        '#dadce0',  # Border gray
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg_main'])
        
        # Core components
        self.detector   = PlateDetector()
        self.ocr        = OCREngine()
        self.blacklist  = BlacklistManager()

        # State
        self.image_path       = ""
        self.plate_images     = []
        self.detection_results = []
        self.current_photo    = None
        self.processing       = False

        self._setup_styles()
        self._build_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_image())
        self.root.bind('<Control-d>', lambda e: self._detect_plates())
        self.root.bind('<Control-b>', lambda e: self._show_debug())

    # ── Style Configuration ──────────────────────────────────────────

    def _setup_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 20, 'bold'),
                       foreground=self.colors['text_primary'],
                       background=self.colors['bg_main'])
        
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 10),
                       foreground=self.colors['text_secondary'],
                       background=self.colors['bg_main'])
        
        style.configure('Card.TFrame',
                       background=self.colors['bg_card'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('Card.TLabelframe',
                       background=self.colors['bg_card'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('Card.TLabelframe.Label',
                       font=('Segoe UI', 11, 'bold'),
                       foreground=self.colors['text_primary'],
                       background=self.colors['bg_card'])
        
        # Button styles
        style.configure('Primary.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       padding=(20, 10))
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary_dark'])])
        
        style.configure('Secondary.TButton',
                       font=('Segoe UI', 10),
                       padding=(20, 10))
        
        style.configure('Success.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       background=self.colors['success'],
                       foreground='white',
                       borderwidth=0,
                       padding=(20, 10))
        
        # Treeview styles
        style.configure('Treeview',
                       font=('Segoe UI', 10),
                       rowheight=35,
                       background=self.colors['bg_card'],
                       fieldbackground=self.colors['bg_card'])
        
        style.configure('Treeview.Heading',
                       font=('Segoe UI', 10, 'bold'),
                       background=self.colors['bg_main'],
                       foreground=self.colors['text_primary'])
        
        # Status bar style
        style.configure('Status.TLabel',
                       font=('Segoe UI', 9),
                       foreground=self.colors['text_secondary'],
                       background=self.colors['bg_card'],
                       padding=(10, 5))

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        """Build the complete UI layout."""
        # Main container
        self.main_container = tk.Frame(self.root, bg=self.colors['bg_main'])
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self._build_header()
        
        # Content area (2 columns on large screens)
        self.content_frame = tk.Frame(self.main_container, bg=self.colors['bg_main'])
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 10))
        
        # Left column - Image and controls
        self.left_column = tk.Frame(self.content_frame, bg=self.colors['bg_main'])
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right column - Results
        self.right_column = tk.Frame(self.content_frame, bg=self.colors['bg_main'])
        self.right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._build_left_panel()
        self._build_right_panel()
        
        # Status bar
        self._build_status_bar()

    def _build_header(self):
        """Build the application header."""
        header_frame = tk.Frame(self.main_container, bg=self.colors['bg_main'])
        header_frame.pack(fill=tk.X)
        
        # Logo and title
        title_frame = tk.Frame(header_frame, bg=self.colors['bg_main'])
        title_frame.pack(side=tk.LEFT)
        
        # App icon (using unicode as placeholder)
        icon_label = tk.Label(title_frame, text="🚗", 
                             font=('Segoe UI', 28),
                             bg=self.colors['bg_main'])
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        title_text = tk.Frame(title_frame, bg=self.colors['bg_main'])
        title_text.pack(side=tk.LEFT)
        
        tk.Label(title_text, text="License Plate",
                font=('Segoe UI', 20, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_main']).pack(anchor=tk.W)
        
        tk.Label(title_text, text="Recognition System",
                font=('Segoe UI', 14),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_main']).pack(anchor=tk.W)
        
        # Quick stats (right side)
        stats_frame = tk.Frame(header_frame, bg=self.colors['bg_main'])
        stats_frame.pack(side=tk.RIGHT)
        
        self.stats_label = tk.Label(stats_frame,
                                   text=f"Blacklist: {len(self.blacklist.df)} entries",
                                   font=('Segoe UI', 9),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['bg_main'])
        self.stats_label.pack()

    def _build_left_panel(self):
        """Build the left panel with image preview and controls."""
        # Action buttons card
        button_card = tk.Frame(self.left_column, bg=self.colors['bg_card'], 
                              relief='flat', bd=0, highlightthickness=1,
                              highlightbackground=self.colors['border'])
        button_card.pack(fill=tk.X, pady=(0, 10))
        
        button_inner = tk.Frame(button_card, bg=self.colors['bg_card'])
        button_inner.pack(padx=20, pady=15, fill=tk.X)
        
        # Button row
        btn_row = tk.Frame(button_inner, bg=self.colors['bg_card'])
        btn_row.pack(fill=tk.X)
        
        self.btn_open = tk.Button(btn_row, text="📁 Open Image",
                                 font=('Segoe UI', 10, 'bold'),
                                 bg=self.colors['primary'],
                                 fg='white',
                                 activebackground=self.colors['primary_dark'],
                                 activeforeground='white',
                                 relief='flat',
                                 cursor='hand2',
                                 command=self._open_image,
                                 padx=20, pady=10,
                                 borderwidth=0)
        self.btn_open.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_detect = tk.Button(btn_row, text="🔍 Detect Plates",
                                   font=('Segoe UI', 10, 'bold'),
                                   bg=self.colors['success'],
                                   fg='white',
                                   activebackground='#0b8043',
                                   activeforeground='white',
                                   relief='flat',
                                   cursor='hand2',
                                   command=self._detect_plates,
                                   state=tk.DISABLED,
                                   padx=20, pady=10,
                                   borderwidth=0)
        self.btn_detect.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_debug = tk.Button(btn_row, text="🔬 Debug View",
                                  font=('Segoe UI', 10),
                                  bg=self.colors['bg_main'],
                                  fg=self.colors['text_primary'],
                                  activebackground='#e8eaed',
                                  activeforeground=self.colors['text_primary'],
                                  relief='flat',
                                  cursor='hand2',
                                  command=self._show_debug,
                                  state=tk.DISABLED,
                                  padx=20, pady=10,
                                  borderwidth=1)
        self.btn_debug.pack(side=tk.LEFT)
        
        # Keyboard shortcuts hint
        shortcuts_frame = tk.Frame(button_inner, bg=self.colors['bg_card'])
        shortcuts_frame.pack(fill=tk.X, pady=(10, 0))
        
        shortcuts_text = "Shortcuts: Ctrl+O Open | Ctrl+D Detect | Ctrl+B Debug"
        tk.Label(shortcuts_frame, text=shortcuts_text,
                font=('Segoe UI', 8),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_card']).pack(anchor=tk.W)
        
        # Image preview card
        self.image_card = tk.Frame(self.left_column, bg=self.colors['bg_card'],
                                  relief='flat', bd=0, highlightthickness=1,
                                  highlightbackground=self.colors['border'])
        self.image_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        card_header = tk.Frame(self.image_card, bg=self.colors['bg_card'])
        card_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(card_header, text="📷 Image Preview",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(side=tk.LEFT)
        
        self.image_info_label = tk.Label(card_header, text="",
                                        font=('Segoe UI', 9),
                                        fg=self.colors['text_secondary'],
                                        bg=self.colors['bg_card'])
        self.image_info_label.pack(side=tk.RIGHT)
        
        # Canvas container with border
        canvas_container = tk.Frame(self.image_card, bg=self.colors['border'])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.canvas = tk.Canvas(canvas_container, bg='#f0f0f0',
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        self.canvas_text = self.canvas.create_text(
            400, 300,  # Will be centered later
            text="📁 Drop image here or click 'Open Image'",
            font=('Segoe UI', 12),
            fill=self.colors['text_secondary'],
            anchor=tk.CENTER
        )
        
        # Bind canvas resize to center placeholder
        self.canvas.bind('<Configure>', self._center_placeholder)

    def _build_right_panel(self):
        """Build the right panel with results."""
        # Results card
        self.results_card = tk.Frame(self.right_column, bg=self.colors['bg_card'],
                                    relief='flat', bd=0, highlightthickness=1,
                                    highlightbackground=self.colors['border'])
        self.results_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header with count
        results_header = tk.Frame(self.results_card, bg=self.colors['bg_card'])
        results_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(results_header, text="📊 Detection Results",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_card']).pack(side=tk.LEFT)
        
        self.results_count_label = tk.Label(results_header, text="",
                                           font=('Segoe UI', 9, 'bold'),
                                           fg=self.colors['primary'],
                                           bg=self.colors['bg_card'])
        self.results_count_label.pack(side=tk.RIGHT)
        
        # Treeview with custom styling
        tree_container = tk.Frame(self.results_card, bg=self.colors['bg_card'])
        tree_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        # Create treeview
        cols = ("#", "Plate Number", "Status", "Details")
        self.tree = ttk.Treeview(tree_container, columns=cols, show="headings", height=10)
        
        # Configure columns
        self.tree.heading("#", text="#")
        self.tree.heading("Plate Number", text="Plate Number")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Details", text="Details")
        
        self.tree.column("#", width=40, anchor='center')
        self.tree.column("Plate Number", width=150)
        self.tree.column("Status", width=100)
        self.tree.column("Details", width=250)
        
        # Scrollbar
        tree_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tags for styling
        self.tree.tag_configure('blacklisted', 
                               background='#fce8e6',
                               foreground='#c5221f')
        self.tree.tag_configure('clean', 
                               background='#e6f4ea',
                               foreground='#137333')
        self.tree.tag_configure('unreadable', 
                               background='#fef7e0',
                               foreground='#b06000')
        
        # Empty state
        self.empty_label = tk.Label(tree_container, 
                                   text="No results yet.\nClick 'Detect Plates' to start.",
                                   font=('Segoe UI', 11),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['bg_card'],
                                   justify=tk.CENTER)
        self.empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _build_status_bar(self):
        """Build the status bar."""
        status_frame = tk.Frame(self.main_container, bg=self.colors['bg_card'],
                               relief='flat', bd=0, highlightthickness=1,
                               highlightbackground=self.colors['border'])
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        status_inner = tk.Frame(status_frame, bg=self.colors['bg_card'])
        status_inner.pack(fill=tk.X, padx=20, pady=8)
        
        # Status indicator (colored dot)
        self.status_indicator = tk.Canvas(status_inner, width=12, height=12,
                                         bg=self.colors['bg_card'],
                                         highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_status_dot('gray')
        
        # Status text
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(status_inner, textvariable=self.status_var,
                                    font=('Segoe UI', 9),
                                    fg=self.colors['text_secondary'],
                                    bg=self.colors['bg_card'])
        self.status_label.pack(side=tk.LEFT)
        
        # Processing indicator (right side)
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(status_inner, textvariable=self.progress_var,
                                      font=('Segoe UI', 9),
                                      fg=self.colors['primary'],
                                      bg=self.colors['bg_card'])
        self.progress_label.pack(side=tk.RIGHT)
        
        # Version info
        tk.Label(status_inner, text="v1.0.0",
                font=('Segoe UI', 8),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_card']).pack(side=tk.RIGHT, padx=(0, 20))

    # ── Button handlers ───────────────────────────────────────────────

    def _open_image(self):
        """Open and display an image."""
        path = filedialog.askopenfilename(
            title="Select Vehicle Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        
        self.image_path = path
        self._display_image()
        self.btn_detect.config(state=tk.NORMAL)
        self.btn_debug.config(state=tk.DISABLED)
        self._clear_results()
        
        # Update status
        self._set_status(f"Loaded: {os.path.basename(path)}", 'blue')
        
        # Update image info
        file_size = os.path.getsize(path) / 1024  # KB
        self.image_info_label.config(text=f"{os.path.basename(path)} • {file_size:.1f} KB")

    def _detect_plates(self):
        """Run plate detection in a separate thread."""
        if not self.image_path or self.processing:
            return
        
        self.processing = True
        self.btn_detect.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        
        # Start progress animation
        self._animate_processing()
        
        # Run detection in thread
        thread = threading.Thread(target=self._process_detection, daemon=True)
        thread.start()

    def _process_detection(self):
        """Process detection in background thread."""
        try:
            self.root.after(0, lambda: self._set_status("🔍 Detecting plates...", 'yellow'))
            self.root.after(0, lambda: self.progress_var.set("Analyzing image..."))
            
            # Detect plates
            self.plate_images = self.detector.detect_plates(self.image_path)
            
            if not self.plate_images:
                self.root.after(0, lambda: self._on_detection_complete(False))
                return
            
            self.root.after(0, lambda: self.progress_var.set(f"Found {len(self.plate_images)} plate(s). Running OCR..."))
            
            # Process each plate
            raw_results = []
            for i, plate_img in enumerate(self.plate_images):
                self.root.after(0, lambda i=i: self.progress_var.set(f"Processing plate {i+1}/{len(self.plate_images)}..."))
                
                try:
                    text = self.ocr.extract_text(plate_img)
                    blacklisted, reason = self.blacklist.check(text)
                    
                    status = ("UNREADABLE" if text in ("UNREADABLE", "ERROR")
                             else "BLACKLISTED" if blacklisted else "CLEAN")
                    
                    if text in ("UNREADABLE", "ERROR"):
                        reason = "Could not extract text clearly"
                    
                    raw_results.append({
                        'index': i + 1,
                        'plate_text': text,
                        'status': status,
                        'reason': reason,
                        'is_blacklisted': blacklisted,
                        'plate_img': plate_img,
                    })
                except Exception as e:
                    raw_results.append({
                        'index': i + 1,
                        'plate_text': "ERROR",
                        'status': "ERROR",
                        'reason': str(e),
                        'is_blacklisted': False,
                        'plate_img': None,
                    })
            
            self.detection_results = raw_results
            self.root.after(0, lambda: self._on_detection_complete(True))
            
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_detection_complete(self, success: bool):
        """Handle detection completion."""
        self.processing = False
        self.btn_detect.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)
        self.progress_var.set("")
        
        if not success:
            messagebox.showinfo("No Plates Found", 
                              "No license plates were detected in the image.\n\nTry using a clearer image with better lighting.")
            self._set_status("No plates detected", 'gray')
            return
        
        # Update results
        unique = self._deduplicate(self.detection_results)
        self._update_results(unique)
        
        # Check for blacklisted plates
        bad = [r for r in unique if r['is_blacklisted']]
        if bad:
            self._show_alert(bad)
            self._set_status(f"⚠️ ALERT: {len(bad)} blacklisted plate(s) found!", 'red')
        else:
            self._set_status(f"✅ Analysis complete — {len(unique)} plate(s) processed successfully", 'green')
        
        self.btn_debug.config(state=tk.NORMAL)

    def _on_error(self, error_msg: str):
        """Handle errors."""
        self.processing = False
        self.btn_detect.config(state=tk.NORMAL)
        self.btn_open.config(state=tk.NORMAL)
        self.progress_var.set("")
        
        messagebox.showerror("Error", f"An error occurred during processing:\n\n{error_msg}")
        self._set_status(f"Error: {error_msg}", 'red')

    def _show_debug(self):
        """Open debug window."""
        if not self.detection_results:
            messagebox.showinfo("No Data", "Please run plate detection first.")
            return
        DebugWindow(self.root, self.detection_results, self.colors)

    # ── UI Update Helpers ────────────────────────────────────────────

    def _display_image(self):
        """Display the loaded image on canvas."""
        try:
            img = Image.open(self.image_path)
            
            # Get canvas size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 600
                canvas_height = 400
            
            # Calculate scaling
            img_width, img_height = img.size
            scale = min(canvas_width/img_width, canvas_height/img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            
            # Clear canvas and display
            self.canvas.delete("all")
            
            # Center image
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display image: {e}")

    def _update_results(self, results: list):
        """Update the results treeview."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Hide empty state
        self.empty_label.place_forget()
        
        # Add results
        for r in results:
            # Determine tag
            if r['is_blacklisted']:
                tag = 'blacklisted'
                status_text = '⚠ BLACKLISTED'
            elif r['status'] == "CLEAN":
                tag = 'clean'
                status_text = '✓ CLEAN'
            else:
                tag = 'unreadable'
                status_text = '? UNREADABLE'
            
            self.tree.insert("", "end",
                           values=(r['index'], r['plate_text'], status_text, r['reason']),
                           tags=(tag,))
        
        # Update count
        self.results_count_label.config(text=f"{len(results)} result(s)")

    def _show_alert(self, blacklisted_plates: list):
        """Show security alert for blacklisted plates."""
        alert_window = tk.Toplevel(self.root)
        alert_window.title("⚠️ Security Alert")
        alert_window.geometry("500x400")
        alert_window.configure(bg='white')
        alert_window.transient(self.root)
        alert_window.grab_set()
        
        # Header
        header = tk.Frame(alert_window, bg='#c5221f', height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="⚠️", font=('Segoe UI', 30),
                bg='#c5221f', fg='white').pack(pady=(10, 0))
        tk.Label(header, text="SECURITY ALERT", font=('Segoe UI', 16, 'bold'),
                bg='#c5221f', fg='white').pack()
        
        # Content
        content = tk.Frame(alert_window, bg='white')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(content, text=f"{len(blacklisted_plates)} Blacklisted Vehicle(s) Detected!",
                font=('Segoe UI', 12, 'bold'),
                fg='#c5221f', bg='white').pack(pady=(0, 20))
        
        # List of plates
        for plate in blacklisted_plates:
            plate_frame = tk.Frame(content, bg='#fce8e6', relief='flat', bd=0)
            plate_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(plate_frame, text=f"  {plate['plate_text']}  ",
                    font=('Segoe UI', 12, 'bold'),
                    fg='#c5221f', bg='#fce8e6').pack(side=tk.LEFT, padx=10, pady=8)
            
            tk.Label(plate_frame, text=plate.get('reason', 'Unknown reason'),
                    font=('Segoe UI', 10),
                    fg='#5f6368', bg='#fce8e6').pack(side=tk.LEFT, padx=5)
        
        # Close button
        tk.Button(content, text="Acknowledge",
                 font=('Segoe UI', 10, 'bold'),
                 bg='#c5221f', fg='white',
                 relief='flat', cursor='hand2',
                 command=alert_window.destroy,
                 padx=30, pady=10).pack(pady=20)

    def _clear_results(self):
        """Clear all results."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.plate_images.clear()
        self.detection_results.clear()
        self.results_count_label.config(text="")
        self.empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _set_status(self, message: str, color: str):
        """Update status bar."""
        self.status_var.set(message)
        self._draw_status_dot(color)

    def _draw_status_dot(self, color: str):
        """Draw colored status indicator."""
        self.status_indicator.delete("all")
        colors = {
            'green': '#0f9d58',
            'red': '#ea4335',
            'yellow': '#f4b400',
            'blue': '#1a73e8',
            'gray': '#9aa0a6'
        }
        dot_color = colors.get(color, colors['gray'])
        
        self.status_indicator.create_oval(2, 2, 10, 10, 
                                         fill=dot_color, outline='')

    def _animate_processing(self):
        """Animate processing indicator."""
        if not self.processing:
            return
        
        current = self.progress_var.get()
        dots = current.count('.')
        if dots >= 3:
            self.progress_var.set("Processing")
        else:
            self.progress_var.set(current + '.')
        
        self.root.after(500, self._animate_processing)

    def _center_placeholder(self, event=None):
        """Center the placeholder text on canvas."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.canvas.coords(self.canvas_text, width//2, height//2)

    def _deduplicate(self, results: list) -> list:
        """Remove duplicate results."""
        seen = {}
        for r in results:
            txt = r['plate_text']
            if txt in ("UNREADABLE", "ERROR"):
                seen.setdefault(txt, r)
            elif txt not in seen:
                seen[txt] = r
            elif r['is_blacklisted'] and not seen[txt]['is_blacklisted']:
                seen[txt] = r
        return list(seen.values())