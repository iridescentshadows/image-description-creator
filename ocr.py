import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import pyperclip
import io
import sys
from paddleocr import PaddleOCR
import re
import numpy as np
import tempfile
import os

class PaddleOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PaddleOCR Image to Text Tool")
        self.root.geometry("900x700")
        
        # Store the current image and OCR results
        self.current_image = None
        self.annotated_image = None
        self.ocr_model = None
        self.current_ocr_result = None
        self.raw_ocr_text = None  # Stores the raw OCR text (newline-joined extracted lines)
        self.formatted_ocr_text = None  # Stores cleaned + paragraph-formatted OCR text

        # Output type for formatting (tweet, tweet thread, etc.)
        self.output_type_var = tk.StringVar(value="tweet")

        # Language options
        self.languages = {
            "Chinese (Simplified)": "ch",
            "English": "en",
            "Japanese": "japan",
            "Korean": "korean",
            "French": "fr",
            "German": "german"
        }
        
        self.setup_ui()
        self.setup_bindings()
        self.init_ocr_model()
        
        self.last_clipboard_image_hash = None
        self.clipboard_monitor_running = False

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control bar
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Language selection
        ttk.Label(control_frame, text="OCR Language:").pack(side=tk.LEFT, padx=(0, 5))
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(control_frame, textvariable=self.lang_var, 
                                   values=list(self.languages.keys()), state="readonly", width=20)
        lang_combo.pack(side=tk.LEFT, padx=(0, 10))
        lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        
        # Instructions label
        instruction_text = "📋 Press Ctrl+V, or Drag & Drop an image file\n"
        instruction_text += "🖼️ Supports: PNG, JPEG, BMP, GIF | Built-in PaddleOCR engine"
        instructions = ttk.Label(control_frame, text=instruction_text, foreground='gray')
        instructions.pack(side=tk.LEFT, padx=(10, 0))
        
        self.auto_clipboard_var = tk.BooleanVar(value=False)
        self.auto_clipboard_cb = ttk.Checkbutton(control_frame, text="Auto-paste from clipboard", 
                                                 variable=self.auto_clipboard_var,
                                                 command=self.toggle_clipboard_monitor)
        self.auto_clipboard_cb.pack(side=tk.LEFT, padx=(10, 0))
        
        self.auto_process_var = tk.BooleanVar(value=False)
        self.auto_process_cb = ttk.Checkbutton(control_frame, text="Auto-process image", 
                                                 variable=self.auto_process_var)
        self.auto_process_cb.pack(side=tk.LEFT, padx=(10, 0))

        # Output type selection
        ttk.Label(control_frame, text="Output:").pack(
            side=tk.LEFT, padx=(10, 5)
        )
        self.output_type_var = tk.StringVar(value="tweet")
        output_combo = ttk.Combobox(
            control_frame,
            textvariable=self.output_type_var,
            values=[
                "tweet",
                "tweet thread",
                "quote retweet",
                "reddit post",
                "reddit comment",
                "reddit thread",
            ],
            state="readonly",
            width=14,
        )
        output_combo.pack(side=tk.LEFT, padx=(0, 5))
        output_combo.bind("<<ComboboxSelected>>", self.on_output_type_change)

        # Paned window for split view
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for image display
        self.image_frame = ttk.LabelFrame(paned, text="Image Preview", padding="5")
        paned.add(self.image_frame, weight=1)
        
        # Image label
        self.image_label = ttk.Label(self.image_frame, text="No image pasted yet\n\nPress Ctrl+V to paste an image")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Right frame for text output
        self.text_frame = ttk.LabelFrame(paned, text="Extracted Text", padding="5")
        paned.add(self.text_frame, weight=1)
        
        # Text widget with scrollbar
        text_container = ttk.Frame(self.text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.text_widget = tk.Text(text_container, wrap=tk.WORD, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.copy_button = ttk.Button(button_frame, text="📋 Copy to Clipboard", 
                                       command=self.copy_to_clipboard, state='disabled')
        self.copy_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="🗑️ Clear All", 
                                        command=self.clear_all)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = ttk.Button(button_frame, text="🔍 Process Image (OCR)", 
                                          command=self.process_image, state='disabled')
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Confidence filter
        ttk.Label(button_frame, text="Min Confidence:").pack(side=tk.LEFT, padx=(20, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(button_frame, from_=0.0, to=1.0, variable=self.confidence_var,
                                      orient=tk.HORIZONTAL, length=100)
        confidence_scale.pack(side=tk.LEFT, padx=(0, 5))
        self.confidence_label = ttk.Label(button_frame, text="0.5")
        self.confidence_label.pack(side=tk.LEFT)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Press Ctrl+V to paste an image")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    def update_confidence_label(self, value):
        """Update confidence label when slider moves"""
        # Fixed: Use config() instead of setText()
        self.confidence_label.config(text=f"{float(value):.2f}")
        
    def setup_bindings(self):
        # Bind Ctrl+V to paste from clipboard
        self.root.bind('<Control-v>', self.paste_from_clipboard)
        
        # Setup Drag and Drop
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)
        
        # Setup hover events for image_label
        self.image_label.bind('<Enter>', self.show_full_image)
        self.image_label.bind('<Leave>', self.hide_full_image)

    def show_full_image(self, event):
        img_source = self.annotated_image if getattr(self, 'annotated_image', None) else self.current_image
        if img_source is None:
            return
            
        self.hover_window = tk.Toplevel(self.root)
        self.hover_window.overrideredirect(True)
        
        x = event.x_root + 15
        y = event.y_root + 15
        
        # Ensure it fits on screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        img_w, img_h = img_source.size
        scale = min((screen_w - x - 20) / img_w, (screen_h - y - 20) / img_h, 1.0)
        
        if scale < 1.0:
            new_size = (int(img_w * scale), int(img_h * scale))
            img_to_show = img_source.resize(new_size, Image.Resampling.LANCZOS)
        else:
            img_to_show = img_source
            
        photo = ImageTk.PhotoImage(img_to_show)
        label = ttk.Label(self.hover_window, image=photo, borderwidth=2, relief="solid")
        label.image = photo
        label.pack()
        self.hover_window.geometry(f"+{x}+{y}")

    def hide_full_image(self, event):
        if hasattr(self, 'hover_window') and self.hover_window:
            self.hover_window.destroy()
            self.hover_window = None

    def handle_drop(self, event):
        file_path = event.data
        # tkinterdnd2 sometimes wraps paths in curly braces if they contain spaces
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        
        try:
            img = Image.open(file_path)
            img.load()  # Ensure it's fully loaded
            self.current_image = img
            self.annotated_image = None
            self.display_image(self.current_image)
            self.status_var.set(f"Image loaded from file. Click 'Process Image' to extract text")
            self.process_button.config(state='normal')
            self.clear_button.config(state='normal')
            if self.auto_process_var.get():
                self.root.after(100, self.process_image)
        except Exception as e:
            self.status_var.set(f"Error loading image from drop: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def toggle_clipboard_monitor(self):
        if self.auto_clipboard_var.get():
            self.clipboard_monitor_running = True
            try:
                clip_img = ImageGrab.grabclipboard()
                if isinstance(clip_img, Image.Image):
                    self.last_clipboard_image_hash = hash(clip_img.tobytes())
            except Exception:
                pass
            self.monitor_clipboard()
        else:
            self.clipboard_monitor_running = False

    def monitor_clipboard(self):
        if not self.clipboard_monitor_running:
            return
        
        try:
            clip_img = ImageGrab.grabclipboard()
            if isinstance(clip_img, Image.Image):
                img_hash = hash(clip_img.tobytes())
                if img_hash != self.last_clipboard_image_hash:
                    self.last_clipboard_image_hash = img_hash
                    self.current_image = clip_img
                    self.annotated_image = None
                    self.display_image(self.current_image)
                    self.status_var.set("Image auto-pasted from clipboard! Click 'Process Image'")
                    self.process_button.config(state='normal')
                    self.clear_button.config(state='normal')
                    if self.auto_process_var.get():
                        self.root.after(100, self.process_image)
        except Exception:
            pass
            
        if self.clipboard_monitor_running:
            self.root.after(1000, self.monitor_clipboard)
        
    def init_ocr_model(self):
        """Initialize PaddleOCR model - runs once at startup"""
        try:
            self.status_var.set("Initializing PaddleOCR model (first time may take a moment)...")
            self.root.update()
            
            # Get selected language code
            lang_code = self.languages[self.lang_var.get()]
            
            # Initialize PaddleOCR
            # use_angle_cls=True enables text direction classification for better accuracy
            self.ocr_model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                lang=lang_code)
            self.status_var.set(f"PaddleOCR ready - Language: {self.lang_var.get()}")
            
        except Exception as e:
            self.status_var.set(f"Failed to initialize PaddleOCR: {str(e)}")
            messagebox.showerror("Init Error", 
                               f"Failed to initialize PaddleOCR.\n\n"
                               f"Make sure you have installed:\n"
                               f"pip install paddlepaddle paddleocr\n\n"
                               f"Error: {str(e)}")
    
    def on_language_change(self, event=None):
        """Reinitialize OCR model when language changes"""
        if self.ocr_model:
            self.init_ocr_model()

    # ---------------------------------------------------------------------------
    #  Paragraph detection helpers (imported from deepseektest.py)
    # ---------------------------------------------------------------------------

    def is_bullet_point(self, text):
        """
        Check if text starts with a bullet point marker.
        Supports: -, *, •, ○, ▪, numbers (1., 2.), letters (a., b.), etc.
        """
        if not text or not text.strip():
            return False
        
        text_stripped = text.lstrip()
        
        # Common bullet point patterns
        bullet_patterns = [
            r'^[-*•○▪►→]',           # Common bullet symbols
            r'^\d+[\.\)]',            # Numbered lists: 1., 2., 1), 2)
            r'^[a-zA-Z][\.\)]',       # Letter lists: a., b., a), b)
            r'^[ivxIVX]+[\.\)]',      # Roman numerals: i., ii., iii.
            r'^[\u2022\u2023\u25E6\u2043\u2219]',  # Unicode bullets
        ]
        
        for pattern in bullet_patterns:
            if re.match(pattern, text_stripped):
                return True
        
        return False

    def detect_list_blocks(self, texts, start_index):
        """
        Detect a consecutive sequence of bullet points starting from start_index.
        Returns the end index of the list (exclusive) or None if not a list.
        """
        if not texts or start_index >= len(texts):
            return None
        
        # Check if current block is a bullet point
        if not self.is_bullet_point(texts[start_index]):
            return None
        
        # Find consecutive bullet points
        end_index = start_index + 1
        while end_index < len(texts) and self.is_bullet_point(texts[end_index]):
            end_index += 1
        
        # Return the range if we have at least 2 bullet points or if it's the only one
        # but we still want to treat single bullet points as lists
        return end_index if end_index > start_index else None

    def detect_paragraph_breaks(self, rec_boxes, texts, line_height_ratio=-0.1):
        """
        Detect whether blocks belong to same paragraph or new paragraph.
        Special handling for bullet points to keep them on separate lines.
        
        Args:
            rec_boxes: List of [x1, y1, x2, y2] coordinates
            texts: List of recognized text strings
            line_height_ratio: Threshold for considering vertical gap as paragraph break
        
        Returns:
            List of integers: 0 = same line (space), 1 = new paragraph (newline),
                             2 = extra new paragraph (double newline)
        """
        if not rec_boxes or len(rec_boxes) == 0:
            return [1]
        
        # Result codes:
        # 0: Same line/paragraph - join with space
        # 1: New paragraph - join with newline
        # 2: Extra new paragraph - join with double newline
        
        result = [1]  # First block always starts a new paragraph
        
        for i in range(1, len(rec_boxes)):
            prev_box = rec_boxes[i-1]
            curr_box = rec_boxes[i]
            prev_text = texts[i-1]
            curr_text = texts[i]
            
            # Check if previous block ends with a Twitter timestamp
            if self.has_twitter_timestamp(prev_text):
                # Force a new paragraph (double newline for separation)
                result.append(2)  # Double newline
                continue
            
            # Check if current block is part of a list
            if self.is_bullet_point(curr_text):
                # Bullet points always get a newline (or double newline based on gap)
                prev_bottom = prev_box[3]
                curr_top = curr_box[1]
                vertical_gap = curr_top - prev_bottom
                curr_height = curr_box[3] - curr_box[1]
                
                # Check if gap is as wide or wider than current block height
                if vertical_gap >= curr_height:
                    result.append(2)  # Double newline for large gaps
                else:
                    result.append(1)  # Single newline for bullet points
                continue
            
            # Normal (non-list) paragraph detection logic
            prev_bottom = prev_box[3]  # y2
            curr_top = curr_box[1]     # y1
            
            # Calculate heights
            prev_height = prev_box[3] - prev_box[1]
            curr_height = curr_box[3] - curr_box[1]
            
            # Calculate vertical gap
            vertical_gap = curr_top - prev_bottom
            
            # Check horizontal overlap (same line/paragraph)
            x_overlap = min(prev_box[2], curr_box[2]) - max(prev_box[0], curr_box[0])
            
            # Determine if same line (horizontal arrangement)
            same_line = abs(curr_top - prev_bottom) < prev_height * 0.3 and x_overlap > 0
            
            if same_line:
                # Same line - definitely same paragraph
                result.append(0)  # Space
            else:
                # Different lines - check the gap size
                # Use double newline only when the gap is significantly larger than
                # the current block height (e.g. between separate tweets in a thread,
                # or between original tweet and quote-retweet comment).
                # A gap equal to block height is just a normal paragraph break
                # (like a blank line between paragraphs within the same tweet).
                if vertical_gap >= curr_height * 2:
                    result.append(2)  # Double newline (large gap = different tweet)
                elif vertical_gap > prev_height * (line_height_ratio + 1):
                    result.append(1)  # Regular new paragraph
                else:
                    result.append(0)  # Same paragraph (but different line - should be space)
        
        return result

    def group_into_paragraphs(self, texts, rec_boxes):
        """
        Group text blocks into paragraphs based on coordinate analysis.
        Special handling for lists to ensure each bullet point is on its own line.
        
        Returns:
            List of paragraphs, where each paragraph is a list of text strings
            and a list of separators between paragraphs
        """
        if not texts or not rec_boxes or len(texts) != len(rec_boxes):
            return [texts] if texts else [], []
        
        # Detect paragraph breaks and gap types
        separators = self.detect_paragraph_breaks(rec_boxes, texts)
        
        # Group into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for i, (text, sep) in enumerate(zip(texts, separators)):
            if i == 0:
                # First block always starts a paragraph
                current_paragraph.append(text)
            elif sep == 0:
                # Same paragraph, add with space
                current_paragraph.append(text)
            else:
                # New paragraph (sep=1 or 2)
                if current_paragraph:
                    paragraphs.append((current_paragraph, separators[i]))
                current_paragraph = [text]
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append((current_paragraph, 1))  # Default separator for last paragraph
        
        return paragraphs

    def format_text_with_paragraphs(self, texts, rec_boxes):
        """
        Format text with proper paragraph detection.
        Joins text within same paragraph with spaces, and paragraphs with newlines.
        Special handling for lists to ensure proper formatting.
        Extra newlines added when gap >= current block height.
        """
        if not texts:
            return ""
        
        # If no boxes available, fall back to simple newline joining
        if not rec_boxes or len(texts) != len(rec_boxes):
            # Still apply list detection even without boxes
            formatted_lines = []
            for text in texts:
                if self.is_bullet_point(text):
                    formatted_lines.append(text)
                else:
                    if formatted_lines and not self.is_bullet_point(formatted_lines[-1]):
                        formatted_lines[-1] += " " + text
                    else:
                        formatted_lines.append(text)
            return "\n".join(formatted_lines)
        
        # Group texts into paragraphs with separators
        paragraphs_with_seps = self.group_into_paragraphs(texts, rec_boxes)
        
        # Join within paragraphs with spaces, between paragraphs with appropriate newlines
        formatted_lines = []
        
        for para_idx, (paragraph, separator) in enumerate(paragraphs_with_seps):
            # Check if this paragraph is a list of bullet points
            if len(paragraph) > 1 and all(self.is_bullet_point(item) for item in paragraph):
                # This is a list - join with newlines instead of spaces
                for bullet_item in paragraph:
                    formatted_lines.append(bullet_item)
            elif len(paragraph) == 1 and self.is_bullet_point(paragraph[0]):
                # Single bullet point - add as its own line
                formatted_lines.append(paragraph[0])
            else:
                # Normal paragraph - join with spaces
                paragraph_text = " ".join(paragraph)
                formatted_lines.append(paragraph_text)
            
            # Add separator between paragraphs (except after the last one)
            if para_idx < len(paragraphs_with_seps) - 1:
                if separator == 2:
                    formatted_lines.append("")  # Double newline
                # separator == 1 gets a single newline (default when joining with \n)
        
        # Join paragraphs with newlines
        return "\n".join(formatted_lines)

    # ---------------------------------------------------------------------------
    #  Handle detection and output formatting
    # ---------------------------------------------------------------------------

    def has_twitter_timestamp(self, text):
        """
        Check if text contains a Twitter-style timestamp.
        Detects patterns like '· 23h', '· 1d', '· 3w', '· Jan 3', '· 14:30',
        '· 26/08/2014', and '23:58·03 Sep 20' (time·DD Mon YY at end of line).
        """
        if not text or not text.strip():
            return False

        stripped = text.strip()

        # Pattern A: Standard format — middle dot BEFORE the time/date component
        # e.g. "· 23h", "· Jan 3", "· 14:30", "· 26/08/2014"
        # Also matches when the dot appears mid-line with text before it,
        # as long as the timestamp component extends to end of line.
        pattern_a = re.compile(
            r"·\s*"
            r"(?:"
            r"\d+[hmdw]"
            r"|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}"
            r"|"
            r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?"
            r"|"
            r"\d{1,2}/\d{1,2}/\d{2,4}"  # date with slashes: 26/08/2014
            r"|"
            r"(?:Just now|Yesterday|Today|\d+\s+(?:min|hour|day|week|month|year)s?\s+ago)"
            r")"
            r"\s*"
            r"(?:[×Xx]|Follow(?:ing)?|Repost(?:ed)?|Like(?:d)?|Reply(?:ing)?|"
            r"\d+(?:\.\d+)?[KkMm]?|@\w+)?"
            r"\s*$",
            re.IGNORECASE,
        )
        if pattern_a.search(stripped):
            return True

        # Pattern B: Time followed by middle dot, then DD Mon YY at end of line
        # e.g. "23:58·03 Sep 20"  (HH:MM·DD Mon YY)
        # This handles the format where the timestamp appears as a suffix on
        # the tweet text line, e.g. "PLEASE respond... 23:58·03 Sep 20"
        pattern_b = re.compile(
            r"\d{1,2}:\d{2}(?::\d{2})?\s*"  # time: HH:MM or HH:MM:SS
            r"·\s*"                            # middle dot separator
            r"\d{1,2}\s+"                      # day number
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"  # month
            r"\s+\d{2,4}"                      # year (2 or 4 digits)
            r"\s*$",
            re.IGNORECASE,
        )
        if pattern_b.search(stripped):
            return True

        return False

    def is_statistics_line(self, text):
        """
        Detect lines that contain only statistics, engagement metrics, view counts,
        dates, or other non-content metadata (superfluous text).

        Detects patterns like:
        - '42 Retweets 4 Quotes 1,567 Likes'
        - '90.3K Views' / '61.8K' / '3.3M' / '80.4K'
        - 'Q281 t16,679' / 'D3 172 1,379 ill 80.4K'
        - '8:44 · 04 Dec 23 · 90.3K Views'
        - '04 Dec 23' (date-only lines)
        - '1.2M' / '16,679' / 't16,679' (pure number/abbreviation lines)
        - '10.3K Retweets and comments 24K Likes' (with 'and' between keywords)
        - '06:15 · 26/08/2014 · Twitter Web Client' (time + slash-date + source)
        """
        if not text or not text.strip():
            return False

        stripped = text.strip()

        # Pattern 1: Engagement keywords with numbers
        # e.g. "42 Retweets 4 Quotes 1,567 Likes"
        # e.g. "90.3K Views" / "1.2M Views"
        # e.g. "10.3K Retweets and comments 24K Likes"
        engagement_keywords = [
            r'Retweets?', r'Quotes?', r'Likes?', r'Views?', r'Reposts?',
            r'Replies?', r'Comments?', r'Shares?', r'Saves?', r'Bookmarks?',
            r'Impressions?', r'Engagements?', r'Followers?', r'Following?',
            r'Subscribers?', r'Liked', r'Reposted', r'Follow(?:ing)?',
        ]
        # Build a pattern that matches a line consisting of numbers + these keywords
        # Allow: numbers (with K/M/B suffixes), commas, dots, the keywords, and
        # the word "and" between keywords (e.g. "10.3K Retweets and comments 24K Likes")
        # Also allow alphabetic words between keywords (e.g. "Quote Tweets" where
        # "Tweets" is an alphabetic word between the "Quote" keyword and the next
        # number/keyword group).
        num_chars = r'[\d,.\sKkMmBbTt' + ''.join(chr(c) for c in range(0x00A0, 0x00C0)) + r']'
        # Between keywords, allow either num_chars sequences OR alphabetic words
        # (2+ letters). This handles cases like "Quote Tweets" where "Tweets" is
        # an alphabetic word between the "Quote" keyword and the next number.
        inter_keyword = r'(?:' + num_chars + r'+|[A-Za-z]{2,}\s*)+'
        engagement_pattern = (
            r'^'
            + num_chars + r'*'
            + r'(?:' + '|'.join(engagement_keywords) + r')'
            + r'(?:' + inter_keyword + r'(?:' + '|'.join(engagement_keywords) + r')' + r')?'
            + r'(?:' + inter_keyword + r'(?:' + '|'.join(engagement_keywords) + r')' + r')?'
            + r'(?:' + inter_keyword + r'(?:' + '|'.join(engagement_keywords) + r')' + r')?'
            + inter_keyword + r'?'
            + r'$'
        )
        if re.match(engagement_pattern, stripped, re.IGNORECASE):
            return True

        # Pattern 1b: Engagement keywords with "and" between them (variant)
        # e.g. "10.3K Retweets and comments 24K Likes"
        # or "10.3K Retweets and comments 24K Likes and 5K Shares"
        # This pattern handles the case where "and" connects two or more keyword phrases.
        # Each "and" segment can have one or more (num*? keyword) pairs.
        # Structure: ^ num*? kw (?: and (?: num*? kw )+ )+ num*? kw? num* $
        and_engagement_pattern = re.compile(
            r'^'
            + num_chars + r'*?'
            + r'(?:' + '|'.join(engagement_keywords) + r')'
            + r'(?:'
            + r'\s+and\s+'
            + r'(?:'
            + num_chars + r'*?'
            + r'(?:' + '|'.join(engagement_keywords) + r')'
            + r')+'
            + r')+'
            + num_chars + r'*?'
            + r'(?:' + '|'.join(engagement_keywords) + r')?'
            + num_chars + r'*'
            + r'$',
            re.IGNORECASE,
        )
        if and_engagement_pattern.match(stripped):
            return True

        # Pattern 2: Date format "04 Dec 23" or "Dec 04, 2023" as the entire line
        date_pattern = re.compile(
            r'^'
            r'(?:'
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}'
            r'|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}'
            r')'
            r'\s*$',
            re.IGNORECASE,
        )
        if date_pattern.match(stripped):
            return True

        # Pattern 3: Time + date + stats combo
        # e.g. "8:44 · 04 Dec 23 · 90.3K Views"
        time_date_stats_pattern = re.compile(
            r'^'
            r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?'  # time
            r'\s*[·\-–]\s*'  # separator
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}'  # date
            r'(?:\s*[·\-–]\s*.+)?'  # optional more stuff after
            r'\s*$',
            re.IGNORECASE,
        )
        if time_date_stats_pattern.match(stripped):
            return True

        # Pattern 4: Time + date with slashes + source text
        # e.g. "06:15 · 26/08/2014 · Twitter Web Client"
        # e.g. "12:30 · 01/01/2023 · Twitter for iPhone"
        # e.g. "8:44 · 04/12/23 · Twitter Web App"
        time_slash_date_source = re.compile(
            r'^'
            r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?'  # time
            r'\s*[·\-–]\s*'  # separator
            r'\d{1,2}/\d{1,2}/\d{2,4}'  # date with slashes (DD/MM/YYYY or MM/DD/YYYY)
            r'(?:\s*[·\-–]\s*'  # separator
            r'.+)?'  # optional source text (e.g. "Twitter Web Client")
            r'\s*$',
            re.IGNORECASE,
        )
        if time_slash_date_source.match(stripped):
            return True

        # Pattern 5: Lines that are mostly numbers/abbreviations with no real words
        # e.g. "Q281 t16,679" "61.8K ill 3.3M 8" "D3 172 1,379 ill 80.4K"
        # These consist of: optional letter prefix + numbers, commas, dots, K/M/B suffixes,
        # and at most 1-2 short "noise" words (like "ill", "t", "Q", "D")
        # Count "real words" (3+ alphabetic chars) vs non-word tokens
        tokens = stripped.split()
        real_word_count = 0
        stat_token_count = 0
        short_word_count = 0  # tracks short (1-3 char) alpha words that may be OCR noise
        for token in tokens:
            # Check if it's a stat token: number-like (with K/M/B, commas, dots)
            if re.match(r'^[A-Za-z]?\d[\d,.]*[KkMmBbTt]?$', token):
                stat_token_count += 1
            # Check if it's a short noise word (1-2 chars, all alpha)
            elif re.match(r'^[A-Za-z]{1,2}$', token):
                stat_token_count += 1
            # Check if it's a pure number with commas
            elif re.match(r'^[\d,]+$', token):
                stat_token_count += 1
            # Check if it's a number+K/M suffix
            elif re.match(r'^\d+(?:\.\d+)?[KkMmBbTt]$', token):
                stat_token_count += 1
            # Check if it's a 3-char alpha word (like "ill", "the", "for", "and")
            # These are borderline — could be noise or content. Track separately.
            elif re.match(r'^[A-Za-z]{3}$', token):
                short_word_count += 1
            # Check if it's a "jumbled" alphanumeric token — OCR noise where letters
            # and digits are intermixed in ways that don't match clean stat patterns.
            # Examples: "ill637" (3 letters + digits), "t2965.6K1ll90K" (complex mix),
            # "1ll90K" (digit + letters + digits + K suffix).
            # These are tokens that contain BOTH letters AND digits but don't match
            # the clean patterns above. They are almost always OCR noise, not real words.
            elif re.match(r'^(?=.*[A-Za-z])(?=.*\d).+$', token):
                stat_token_count += 1
            # Otherwise it's a real word (4+ chars or contains non-alpha chars)
            else:
                real_word_count += 1

        # If line has at least 2 tokens and >80% are stat tokens, it's a stat line
        if len(tokens) >= 2 and stat_token_count > 0 and real_word_count == 0:
            return True
        # Also catch lines with 1 real word and rest stats (e.g. "61.8K ill 3.3M 8")
        if len(tokens) >= 3 and stat_token_count >= 2 and real_word_count <= 1:
            return True
        # Catch 2-token lines like "ill 3.3M" where one token is a stat number
        # and the other is a short (1-3 char) word that looks like OCR noise.
        # The short word must be 3 chars or fewer, and there must be at least
        # one stat number token present.
        if len(tokens) >= 2 and stat_token_count >= 1 and real_word_count == 0 and short_word_count >= 1:
            return True

        # Pattern 6: Single token that is just a number/abbreviation (e.g. "16,679", "61.8K")
        # Only flag if it looks like a stat and is on its own line
        if len(tokens) == 1:
            single = tokens[0]
            # Pure number with optional commas (any length, including single digits like "8")
            if re.match(r'^[\d,]+$', single):
                return True
            # Number with K/M/B suffix
            if re.match(r'^\d+(?:\.\d+)?[KkMmBbTt]$', single):
                return True
            # Letter + number combo like "Q281", "D3", "t16,679"
            if re.match(r'^[A-Za-z]\d[\d,]*[KkMmBbTt]?$', single):
                return True
            # Jumbled alphanumeric token like "ill637" (letters + digits mixed)
            if re.match(r'^(?=.*[A-Za-z])(?=.*\d).+$', single):
                return True
            # Single letter (1 char, alpha only) — OCR noise on its own line
            if re.match(r'^[A-Za-z]$', single):
                return True

        return False

    def strip_statistics(self, text):
        """Remove lines that contain only statistics/engagement/date metadata."""
        if not text:
            return text
        lines = text.split("\n")
        cleaned = [
            line
            for line in lines
            if not self.is_statistics_line(line.strip())
        ]
        return "\n".join(cleaned)

    def detect_handles(self, text):
        """
        Scan OCR text for Twitter handles (@username).
        Removes lines containing handles and strips the nickname text immediately
        before a handle (e.g. 'John Doe @johndoe'), since that text is usually the
        poster's display name.

        Also handles the case where OCR splits nickname and handle across two lines:
            John Doe
            @johndoe
        -> both the nickname line and the @handle line are removed.

        Also handles lines where the handle is followed by a timestamp or other
        metadata (e.g. "JD Schooley @DirtyDog650 · 23h"):
        -> extracts @DirtyDog650 and removes the entire line.

        Also strips stray single-character lines that appear immediately before
        a handle line (e.g. a stray "A" from OCR misreading the X.com logo or
        avatar), since those are typically OCR noise, not tweet content.

        Returns: (handles_list, cleaned_text)
        """
        if not text:
            return [], text

        lines = text.split("\n")
        handles = []
        cleaned_lines = []
        skip_next = False  # flag to skip a line that is a nickname for the next @handle line

        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue

            stripped = line.strip()

            # Case 1: nickname and handle on the same line, e.g. "John Doe @johndoe"
            # or "JD Schooley @DirtyDog650 · 23h" (handle followed by timestamp/metadata)
            # or "Oklahoma Department of... @OKWildlifeDept" (long display name + handle)
            # Match: optional nickname (0-50 chars), then @handle, then optionally
            # more text (timestamp, etc.). Using 50 chars to accommodate display names
            # that OCR may place before @handle.
            match = re.match(r"^(.{0,50}?)@([\w.]+)", stripped)
            if match:
                # Check that the handle appears early enough in the line to be a
                # poster line, not a mid-tweet mention. If there's significant text
                # AFTER the handle that looks like tweet content (not just metadata),
                # this might be a false positive. But for now, the 20-char nickname
                # limit is a reasonable heuristic.
                handle = "@" + match.group(2)
                handles.append(handle)
                # Remove the entire line (nickname + handle + any trailing text)
                # Also remove any stray single-character line immediately above
                # this handle line (e.g. "A" from OCR misreading the X.com logo).
                if cleaned_lines and len(cleaned_lines[-1].strip()) == 1:
                    cleaned_lines.pop()
                continue

            # Case 2: this line is just a @handle by itself
            handle_match = re.match(r"^@([\w.]+)$", stripped)
            if handle_match:
                handle = "@" + handle_match.group(1)
                handles.append(handle)
                # Remove the handle line from the body
                # Also remove any stray single-character line immediately above
                # this handle line (e.g. "A" from OCR misreading the X.com logo).
                if cleaned_lines and len(cleaned_lines[-1].strip()) == 1:
                    cleaned_lines.pop()
                continue

            # Case 3: this line might be a nickname, and the next line is @handle
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                next_handle = re.match(r"^@([\w.]+)$", next_stripped)
                if next_handle:
                    # Current line is a nickname, next line is the handle
                    handle = "@" + next_handle.group(1)
                    handles.append(handle)
                    # Skip both the nickname line and the handle line
                    skip_next = True
                    # Also remove any stray single-character line immediately above
                    # the nickname line (e.g. "A" from OCR misreading the X.com logo).
                    if cleaned_lines and len(cleaned_lines[-1].strip()) == 1:
                        cleaned_lines.pop()
                    continue

            # Not a handle-related line, keep as-is
            cleaned_lines.append(line)

        return handles, "\n".join(cleaned_lines)

    def strip_reply_to(self, text):
        """
        Remove 'Replying to @handle' prefixes from lines, keeping the rest
        of the content. Also handles '@handle Replying to @handle' patterns
        that appear mid-line (which indicate a separate tweet embedded in the
        same OCR line), splitting the line at that boundary.
        """
        if not text:
            return text
        lines = text.split("\n")
        # Pattern for "Replying to @handle" at the start of a line
        reply_prefix_pattern = re.compile(
            r'^(Replying\s+to\s+@[\w.]+)\s*',
            re.IGNORECASE,
        )
        # Pattern for "@handle Replying to @handle" mid-line — this indicates
        # a new tweet starts here. We split the line at this point.
        # e.g. "...text. @OKWildlifeDept Replying to @DirtyDog650 fish is..."
        # We capture the first @handle separately so it can be preserved as
        # a handle line for the new tweet section.
        mid_reply_pattern = re.compile(
            r'(@[\w.]+)\s+Replying\s+to\s+@[\w.]+',
            re.IGNORECASE,
        )
        # Pattern for trailing display name before a mid-line reply marker.
        # OCR often places the display name right before the @handle, e.g.
        # "...Oklahoma Department of...   @OKWildlifeDept Replying to..."
        # This pattern matches text ending with "..." that consists only of
        # alphabetic words (at least 2 alpha chars, no numbers), to avoid
        # matching stat fragments like "80.4K" where "K" could be mistaken.
        trailing_display_name = re.compile(
            r'\s*[A-Za-z]{2,}[A-Za-z\s]*\.\.\.\s*$',
        )
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue
            # Strip "Replying to @handle" prefix
            stripped = reply_prefix_pattern.sub('', stripped)
            stripped = stripped.strip()
            if not stripped:
                continue
            # Check for mid-line "Replying to" patterns and split
            # We split into parts: [text_before, "@handle", text_after]
            # The captured @handle is preserved as a handle line for the
            # new tweet section that follows.
            parts = mid_reply_pattern.split(stripped, maxsplit=1)
            if len(parts) >= 3:
                # parts[0] = text before the reply marker
                # parts[1] = the @handle of the replying account (preserved)
                # parts[2] = text after the reply marker (new tweet content)
                before = parts[0].strip()
                handle = parts[1].strip()
                after = parts[2].strip()

                # Determine if the text before the marker is a display name
                # (belonging to the replying account) rather than tweet content.
                # Two cases:
                #   1. Display name ending with "..." (e.g. "Oklahoma Department of...")
                #   2. Short display name without "..." (e.g. "JD Schooley")
                #      Heuristic: ≤ 50 chars, no sentence-ending punctuation,
                #      and the handle line starts a new tweet (after has content).
                dn_match = trailing_display_name.search(before)
                is_short_display_name = (
                    before
                    and len(before) <= 50
                    and not re.search(r'[.?!]\s*$', before)
                    and after.strip()
                )
                if dn_match or is_short_display_name:
                    if dn_match:
                        display_name = dn_match.group(0).strip()
                        before = before[:dn_match.start()].strip()
                    else:
                        display_name = before
                        before = ""
                    # Prepend the display name to the handle line
                    handle = f"{display_name} {handle}".strip()

                if before:
                    cleaned.append(before)
                if handle:
                    cleaned.append(handle)
                if after:
                    cleaned.append(after)
            else:
                # No mid-line reply found, keep as-is
                cleaned.append(stripped)
        return "\n".join(cleaned)

    def strip_timestamps(self, text):
        """
        Remove or strip Twitter-style timestamps from lines.
        
        If a line consists entirely of a timestamp (or timestamp + metadata),
        the entire line is removed. If a timestamp is embedded within a line
        of text (e.g. "PLEASE respond... 23:58·03 Sep 20"), only the timestamp
        portion is stripped from the line, preserving the surrounding text.
        """
        if not text:
            return text
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue
            
            # Check if the line is purely a timestamp line (remove entirely)
            if self.has_twitter_timestamp(stripped):
                # But first check if there's text before the timestamp
                # that should be preserved
                # Try pattern B first: HH:MM·DD Mon YY at end of line
                ts_pattern_b = re.compile(
                    r"\s*\d{1,2}:\d{2}(?::\d{2})?\s*"
                    r"·\s*"
                    r"\d{1,2}\s+"
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                    r"\s+\d{2,4}"
                    r"\s*$",
                    re.IGNORECASE,
                )
                text_before_b = ts_pattern_b.sub('', stripped).strip()
                if text_before_b != stripped.strip():
                    # Pattern B matched and removed something
                    if text_before_b:
                        # There's text before the timestamp — keep it
                        cleaned.append(text_before_b)
                    # If empty, it was a pure timestamp line — skip it
                    continue
                # Try pattern A: standard ·timestamp format
                ts_pattern_a = re.compile(
                    r"\s*·\s*"
                    r"(?:"
                    r"\d+[hmdw]"
                    r"|"
                    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}"
                    r"|"
                    r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?"
                    r"|"
                    r"\d{1,2}/\d{1,2}/\d{2,4}"
                    r"|"
                    r"(?:Just now|Yesterday|Today|\d+\s+(?:min|hour|day|week|month|year)s?\s+ago)"
                    r")"
                    r"\s*"
                    r"(?:[×Xx]|Follow(?:ing)?|Repost(ed)?|Like(?:d)?|Reply(?:ing)?|"
                    r"\d+(?:\.\d+)?[KkMm]?|@\w+)?"
                    r"\s*$",
                    re.IGNORECASE,
                )
                text_before_a = ts_pattern_a.sub('', stripped).strip()
                if text_before_a != stripped.strip():
                    # Pattern A matched and removed something
                    if text_before_a:
                        cleaned.append(text_before_a)
                    # If empty, it was a pure timestamp line — skip it
                    continue
                # Neither pattern matched (shouldn't happen if has_twitter_timestamp returned True)
                cleaned.append(line)
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def strip_inline_stats(self, text):
        """
        Remove inline statistics and timestamp patterns that appear within
        tweet content lines (not just on their own lines). This handles cases
        where OCR merges stats onto the same line as tweet text.

        Removes patterns like:
        - "D3 172 1,379 ill 80.4K" (stat token sequences)
        - "8:44 · 04 Dec 23 · 90.3K Views 42 Retweets 4 Quotes 1,567 Likes"
          (timestamp + stats combos at end of line)
        """
        if not text:
            return text

        lines = text.split("\n")
        cleaned_lines = []

        # Pattern for inline timestamp+stats combos at end of line:
        # e.g. "text 8:44 · 04 Dec 23 · 90.3K Views 42 Retweets 4 Quotes 1,567 Likes"
        # e.g. "text 06:15 · 26/08/2014 · Twitter Web Client"
        inline_timestamp_stats = re.compile(
            r'\s*\d{1,2}:\d{2}(?::\d{2})?\s*[·\-–]\s*'
            r'(?:'
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}'  # 04 Dec 23
            r'|'
            r'\d{1,2}/\d{1,2}/\d{2,4}'  # 26/08/2014
            r')'
            r'(?:\s*[·\-–]\s*'
            r'[\d,.\sKkMmBbTtA-Za-z]+'
            r'(?:Retweets?|Quotes?|Likes?|Views?|Reposts?|Replies?|Comments?|Shares?|Saves?|Bookmarks?|Impressions?|Engagements?|Followers?|Following?|Subscribers?|Liked|Reposted|Follow)?'
            r'[\d,.\sKkMmBbTtA-Za-z]*)*'
            r'\s*$',
            re.IGNORECASE,
        )

        # Pattern for stat token sequences within a line:
        # e.g. "text. D3 172 1,379 ill 80.4K more text"
        # Matches a sequence of stat tokens: optional-letter+number, short words (1-2 chars),
        # or pure numbers with commas. Each token in the sequence must be a stat-like token.
        inline_stat_sequence = re.compile(
            r'\s+'
            r'(?:'
            r'[A-Za-z]?\d[\d,.]*[KkMmBbTt]?'  # e.g. D3, 172, 1,379, 80.4K
            r'|'
            r'(?=[A-Za-z]*\d)[A-Za-z\d]{2,}'  # jumbled alphanumeric like "ill637", "1ll90K"
            r')'
            r'(?:\s+(?:'
            r'[A-Za-z]?\d[\d,.]*[KkMmBbTt]?'  # more number-like tokens
            r'|'
            r'[A-Za-z]{1,3}(?=\s+(?:[A-Za-z]?\d|[\d,]))'  # short word before more stats
            r'|'
            r'(?=[A-Za-z]*\d)[A-Za-z\d]{2,}'  # jumbled alphanumeric
            r'))*'
            r'\s*',
        )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue

            # First, try to strip inline timestamp+stats combo from the end
            stripped = inline_timestamp_stats.sub('', stripped)

            # Then, try to strip stat sequences from within the line
            # We do this iteratively to handle multiple stat sequences
            prev = None
            while prev != stripped:
                prev = stripped
                stripped = inline_stat_sequence.sub(' ', stripped).strip()

            stripped = stripped.strip()
            if stripped:
                cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def split_into_tweet_chunks(self, text, max_chars=280):
        """
        Split text into chunks of max_chars, breaking at sentence boundaries
        (period, newline, etc.) when possible.
        """
        if not text:
            return []

        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) + 2 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", chunk)
                temp = ""
                for sent in sentences:
                    if len(temp) + len(sent) + 1 > max_chars and temp:
                        final_chunks.append(temp.strip())
                        temp = sent
                    else:
                        if temp:
                            temp += " " + sent
                        else:
                            temp = sent
                if temp:
                    final_chunks.append(temp.strip())

        return final_chunks

    def format_as_tweet(self, text):
        """
        Format as: 'tweet by @handle that says [body]'
        Strips timestamps, statistics, and handle lines from the body.
        """
        if not text:
            return text

        cleaned = self.strip_reply_to(text)
        # Extract handles BEFORE stripping timestamps, since the handle
        # may be on the same line as the timestamp
        handles, cleaned = self.detect_handles(cleaned)
        # Strip timestamps from the body (timestamps are preserved in
        # formatted_ocr_text so earlier formatters can use them for
        # structural detection)
        cleaned = self.strip_timestamps(cleaned)
        cleaned = cleaned.strip()

        handle_str = handles[0] if handles else "@unknown"
        return f"tweet by {handle_str} that says \n{cleaned}"

    def format_as_tweet_thread(self, text):
        """
        Format as:
        tweet thread that goes as follows

        @handle1:
        > [chunk1]

        @handle2:
        > [chunk2]
        ...

        Uses handle lines found in the raw text to identify individual
        tweets in the thread. A "handle line" is a line containing an
        @username with a short nickname prefix (e.g. "John Doe @johndoe"
        or "JD Schooley @DirtyDog650 · 23h"). Each tweet section runs
        from one handle line to the next handle line (or end of text).

        If no handle lines are found, falls back to timestamp-based
        splitting.
        """
        if not text:
            return text

        # First strip "Replying to @handle" prefixes which are metadata
        text = self.strip_reply_to(text)

        lines = text.split("\n")

        # Strategy: find handle lines in the raw text. Each tweet in a
        # thread screenshot starts with a handle line like:
        #   "Oklahoma Department...· 23h"  (display name + timestamp, no @handle visible)
        #   "JD Schooley @DirtyDog650 · 23h"  (nickname + @handle + timestamp)
        #
        # We use these handle lines as delimiters to split the thread
        # into individual tweets. The text before the first handle line
        # is treated as the first tweet (which may not have a detectable
        # @handle in the OCR output).

        # Find all handle line indices.
        # A handle line is one that contains @username with a short
        # nickname prefix (≤20 chars before the @).
        handle_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match: optional short nickname (0-50 chars), then @handle,
            # optionally followed by more text (timestamp, etc.)
            # Using 50 chars to accommodate display names like
            # "Oklahoma Department of..." that OCR may place before @handle.
            if re.match(r"^.{0,50}?@([\w.]+)", stripped):
                handle_indices.append(i)

        if len(handle_indices) >= 1:
            # Split the thread into individual tweets using handle lines
            # as delimiters.
            tweet_sections = []

            # Text before the first handle line is the first tweet section
            # (the first tweet's handle may not have been captured by OCR)
            first_section_lines = lines[:handle_indices[0]]
            if first_section_lines:
                tweet_sections.append(first_section_lines)

            # Each handle line starts a new tweet section
            for idx, h_idx in enumerate(handle_indices):
                if idx + 1 < len(handle_indices):
                    end_idx = handle_indices[idx + 1]
                else:
                    end_idx = len(lines)
                tweet_lines = lines[h_idx:end_idx]
                tweet_sections.append(tweet_lines)

            # Process each tweet section
            chunks = []
            section_handles = []

            for tweet_lines in tweet_sections:
                tweet_raw = "\n".join(tweet_lines)

                # Extract handles BEFORE stripping timestamps, since the
                # handle may be on the same line as the timestamp
                # (e.g. "JD Schooley @DirtyDog650 · 23h")
                tweet_handles, tweet_without_handles = self.detect_handles(tweet_raw)

                # Strip leading display name fragments that OCR merged onto
                # the same line as tweet content. This handles cases like:
                #   "Oklahoma Department... Getting her an engagement ring..."
                # where "Oklahoma Department..." is a display name (ending
                # with "...") that got concatenated with the tweet body.
                # Also handles cases where the display name is followed by a
                # timestamp on the same line (e.g. "Oklahoma Department...· h")
                # with the actual tweet content on the next line.
                tweet_lines_clean = tweet_without_handles.split("\n")
                if tweet_lines_clean:
                    first_line = tweet_lines_clean[0]
                    # Match: text ending with "..." optionally followed by a
                    # timestamp, then either more content on the same line or
                    # nothing (content is on the next line).
                    # Pattern 1: "..." followed by content on the same line
                    # e.g. "Oklahoma Department... Getting her an engagement ring..."
                    dn_content_match = re.match(
                        r'^(.{3,}?\.\.\.)\s+(.+)',
                        first_line,
                    )
                    if dn_content_match:
                        # The part before "..." is a display name fragment
                        # Keep only the content after it
                        tweet_lines_clean[0] = dn_content_match.group(2)
                    else:
                        # Pattern 2: "..." optionally followed by a timestamp,
                        # with no tweet content on this line (content is below).
                        # e.g. "Oklahoma Department...· h" or just "Oklahoma Department..."
                        # Strip this entire line since it's a display name fragment.
                        dn_ts_match = re.match(
                            r'^.{3,}\.\.\.',
                            first_line,
                        )
                        if dn_ts_match:
                            # Remove the display name line entirely
                            tweet_lines_clean.pop(0)

                tweet_cleaned = "\n".join(tweet_lines_clean)

                # Clean the tweet text
                tweet_cleaned = self.strip_inline_stats(tweet_cleaned)
                # Strip timestamps from each tweet chunk (timestamps are preserved
                # in formatted_ocr_text so the handle-based split can use them)
                tweet_cleaned = self.strip_timestamps(tweet_cleaned)
                tweet_cleaned = tweet_cleaned.strip()

                if tweet_cleaned:
                    chunks.append(tweet_cleaned)
                    if tweet_handles:
                        section_handles.append(tweet_handles[0])
                    else:
                        section_handles.append(None)

            if not chunks:
                return text

            # Build the output
            lines_out = ["tweet thread that goes as follows", ""]
            for i, chunk in enumerate(chunks):
                # Use the handle found in this section, or fall back to @unknown
                handle = section_handles[i] if i < len(section_handles) and section_handles[i] else "@unknown"
                lines_out.append(f"{handle}:")
                lines_out.append(f"> {chunk}")
                if i < len(chunks) - 1:
                    lines_out.append("")

            return "\n".join(lines_out)
        else:
            # No handle lines found — fall back to timestamp-based splitting
            timestamp_indices = []
            for i, line in enumerate(lines):
                if self.has_twitter_timestamp(line):
                    timestamp_indices.append(i)

            if len(timestamp_indices) >= 1:
                # Split using timestamps as delimiters
                tweet_sections = []
                processed_up_to = 0

                for ts_idx in timestamp_indices:
                    relative_ts = ts_idx - processed_up_to

                    section_start = None
                    for j in range(relative_ts - 1, -1, -1):
                        stripped = lines[j].strip()
                        if re.match(r"^@([\w.]+)$", stripped):
                            section_start = j
                            break
                        if not stripped:
                            found_handle_above = False
                            for k in range(j - 1, -1, -1):
                                above = lines[k].strip()
                                if re.match(r"^@([\w.]+)$", above):
                                    section_start = j + 1
                                    found_handle_above = True
                                    break
                                if above:
                                    break
                            if found_handle_above:
                                break
                            continue

                    if section_start is None:
                        section_start = 0

                    tweet_lines = lines[section_start:relative_ts + 1]
                    tweet_sections.append(tweet_lines)

                    lines = lines[relative_ts + 1:]
                    processed_up_to = ts_idx + 1

                chunks = []
                section_handles = []

                for tweet_lines in tweet_sections:
                    tweet_raw = "\n".join(tweet_lines)
                    tweet_handles, tweet_without_handles = self.detect_handles(tweet_raw)
                    # Strip timestamps from each tweet chunk (timestamps were used
                    # for splitting above, now remove them from the output)
                    tweet_cleaned = self.strip_timestamps(tweet_without_handles)
                    tweet_cleaned = tweet_cleaned.strip()

                    if tweet_cleaned:
                        chunks.append(tweet_cleaned)
                        if tweet_handles:
                            section_handles.append(tweet_handles[0])
                        else:
                            section_handles.append(None)

                if not chunks:
                    return text

                lines_out = ["tweet thread that goes as follows", ""]
                for i, chunk in enumerate(chunks):
                    handle = section_handles[i] if i < len(section_handles) and section_handles[i] else "@unknown"
                    lines_out.append(f"{handle}:")
                    lines_out.append(f"> {chunk}")
                    if i < len(chunks) - 1:
                        lines_out.append("")

                return "\n".join(lines_out)
            else:
                # No timestamps found — fall back to content-based splitting
                handles, cleaned = self.detect_handles(text)
                # Strip timestamps from the text before content-based splitting
                cleaned = self.strip_timestamps(cleaned)
                cleaned = cleaned.strip()

                chunks = self.split_into_tweet_chunks(cleaned, 280)

                if not chunks:
                    return text

                lines_out = ["tweet thread that goes as follows", ""]
                for i, chunk in enumerate(chunks):
                    handle = handles[i] if i < len(handles) else (
                        handles[-1] if handles else "@unknown"
                    )
                    lines_out.append(f"{handle}:")
                    lines_out.append(f"> {chunk}")
                    if i < len(chunks) - 1:
                        lines_out.append("")

                return "\n".join(lines_out)

    def format_as_quote_retweet(self, text):
        """
        Format as:
        quote retweet. the original tweet is by @handle_a and says
        > [original_text]. @handle_b then quote retweets this and says
        > [comment_text]

        Structure of a quote-retweet screenshot:
          [Quote retweeter's display name + @handle]
          [Quote retweeter's comment text]
          [· 23h  <-- timestamp of the quote retweet]
          [divider / blank line]
          [Embedded quoted tweet card:]
            [Original poster's display name + @handle]
            [Original tweet body]
            [· 23h  <-- timestamp of the original tweet (if visible)]
            [stats line]

        The timestamp belongs to the quote retweet (comment), not the
        original tweet. Everything from the start up to and including
        the timestamp line is the comment section. Everything after the
        blank line separator is the original tweet section.
        """
        if not text:
            return text

        # First strip "Replying to @handle" lines which are metadata
        text = self.strip_reply_to(text)

        # Work with the raw text to find structural landmarks before cleaning.
        lines = text.split("\n")

        # Strategy: find the LAST timestamp line in the raw text.
        # In a quote-retweet screenshot, the timestamp appears at the
        # bottom of the quote retweet (comment) section. The original
        # tweet is below the blank line separator.
        #
        # The timestamp marks the end of the COMMENT section, not the
        # original tweet. Everything from the start up to the timestamp
        # line is the comment. Everything after the blank line separator
        # below the timestamp is the original tweet.

        # Find the last timestamp line in the raw text.
        timestamp_line_idx = None
        for i, line in enumerate(lines):
            if self.has_twitter_timestamp(line):
                timestamp_line_idx = i

        if timestamp_line_idx is not None:
            # The COMMENT section ends at the timestamp line.
            # The comment is everything from the start to the timestamp.
            # The original tweet is everything after the blank line
            # separator below the timestamp.

            # Find the first non-blank line AFTER the timestamp line —
            # that's the start of the original tweet section.
            original_start = None
            for j in range(timestamp_line_idx + 1, len(lines)):
                if lines[j].strip():
                    original_start = j
                    break

            if original_start is not None:
                # Comment = lines from start to timestamp (inclusive)
                comment_lines = lines[:timestamp_line_idx + 1]
                # Original = lines from original_start to end
                original_lines = lines[original_start:]
            else:
                # No content found after timestamp — fallback
                comment_lines = lines[:timestamp_line_idx + 1]
                original_lines = []
        else:
            # No timestamp found — fall back to the original double-newline split
            handles, cleaned = self.detect_handles(text)
            # Strip timestamps from the text before splitting
            cleaned = self.strip_timestamps(cleaned)
            cleaned = cleaned.strip()

            parts = re.split(r"\n\n+", cleaned, maxsplit=1)
            if len(parts) >= 2:
                original_text = parts[0].strip()
                comment_text = parts[1].strip()
            else:
                mid = len(cleaned) // 2
                split_pos = cleaned.rfind("\n\n", 0, mid)
                if split_pos == -1:
                    split_pos = cleaned.rfind(". ", 0, mid)
                    if split_pos != -1:
                        split_pos += 1
                    else:
                        split_pos = mid
                original_text = cleaned[:split_pos].strip()
                comment_text = cleaned[split_pos:].strip()

            # handles are in visual order: first = quote retweeter, last = original author
            if len(handles) >= 2:
                handle_a = handles[-1]  # Last = original author
                handle_b = handles[0]   # First = quote retweeter
            elif len(handles) == 1:
                handle_a = "@original"
                handle_b = handles[0]
            else:
                handle_a = "@original"
                handle_b = "@commenter"
            return (
                f"quote retweet. the original tweet is by {handle_a} and says\n\n"
                f"> {original_text}\n\n"
                f"{handle_b} then quote retweets this and says\n"
                f"> {comment_text}"
            )

        # Clean the extracted sections
        comment_raw = "\n".join(comment_lines)
        original_raw = "\n".join(original_lines)

        # --- Clean the COMMENT section ---
        # The comment section may have the display name + @handle + timestamp
        # all merged into one line by OCR. Handle this by stripping the
        # leading display name + @handle prefix and trailing timestamp inline.
        if timestamp_line_idx is not None and len(comment_lines) == 1:
            comment_line = comment_lines[0]
            # Extract handles for the output
            handle_match = re.match(r"^(.{0,50}?)@([\w.]+)\s*", comment_line)
            if handle_match:
                comment_handles = ["@" + handle_match.group(2)]
                # Remove the leading display name + handle prefix
                body_start = handle_match.end()
                comment_cleaned = comment_line[body_start:].strip()
            else:
                comment_handles = []
                comment_cleaned = comment_line
            # Strip trailing timestamp from the end of the body inline
            ts_pattern = re.compile(
                r"\s*·\s+\d+[hmdw]\s*$",
                re.IGNORECASE,
            )
            comment_cleaned = ts_pattern.sub('', comment_cleaned)
            # Also strip any trailing display name + @handle duplication
            # that OCR may have merged onto the same line
            comment_cleaned = re.sub(
                r'\s*\w+\s+@[\w.]+\s*$',
                '',
                comment_cleaned,
            ).strip()
            comment_cleaned = comment_cleaned.strip()
        else:
            comment_handles, comment_without_handles = self.detect_handles(comment_raw)
            # Strip timestamps from the comment section (timestamps were used
            # for structural splitting above, now remove them from the output)
            comment_cleaned = self.strip_timestamps(comment_without_handles)
            comment_cleaned = comment_cleaned.strip()

            # If detect_handles() wiped out the entire comment (false positive
            # where an @mention in the content was mistaken for a poster handle),
            # fall back to just stripping the @mention(s) inline instead.
            if not comment_cleaned and comment_raw.strip():
                comment_cleaned = re.sub(
                    r'@([\w.]+)',
                    r'@/\1',
                    comment_raw,
                ).strip()
                comment_handles = re.findall(r'@([\w.]+)', comment_raw)
                comment_handles = ['@' + h for h in comment_handles]
                # Also strip timestamps from the fallback
                comment_cleaned = self.strip_timestamps(comment_cleaned)
                comment_cleaned = comment_cleaned.strip()

        # --- Clean the ORIGINAL TWEET section ---
        # The original tweet may be on a single line with an @handle mention
        # as content (e.g. "#nw first watch w/ @SILNTKLL"). detect_handles()
        # would incorrectly remove the entire line. Instead, extract all
        # @handles and strip them inline, keeping the body text.
        #
        # IMPORTANT: When the original tweet section has only 1 line, the
        # handles extracted via re.findall() are CONTENT MENTIONS, not poster
        # handles. The actual original author's handle is typically found in
        # the comment section (on the embedded tweet card header). We track
        # whether original_handles came from content mentions vs poster handles
        # so the handle determination logic below can make the right choice.
        original_handles_are_mentions = False
        if len(original_lines) == 1:
            original_line = original_lines[0]
            # Extract all @handles from the line
            found_handles = re.findall(r'@([\w.]+)', original_line)
            original_handles = ['@' + h for h in found_handles]
            original_handles_are_mentions = True
            # Replace @mentions with @/mention in the body (preserve the text,
            # just change the @ prefix to @/ to avoid confusion with handles)
            original_cleaned = re.sub(r'@([\w.]+)', r'@/\1', original_line).strip()
            # Strip timestamps from the original tweet body
            original_cleaned = self.strip_timestamps(original_cleaned)
            original_cleaned = original_cleaned.strip()
        else:
            original_handles, original_without_handles = self.detect_handles(original_raw)
            # Strip timestamps from the original tweet section (timestamps were
            # used for structural splitting above, now remove them from the output)
            original_cleaned = self.strip_timestamps(original_without_handles)
            original_cleaned = original_cleaned.strip()

            # If detect_handles() wiped out the entire original tweet,
            # fall back to inline @mention stripping
            if not original_cleaned and original_raw.strip():
                original_cleaned = re.sub(
                    r'@([\w.]+)',
                    r'@/\1',
                    original_raw,
                ).strip()
                original_handles = re.findall(r'@([\w.]+)', original_raw)
                original_handles = ['@' + h for h in original_handles]
                original_handles_are_mentions = True
                # Also strip timestamps from the fallback
                original_cleaned = self.strip_timestamps(original_cleaned)
                original_cleaned = original_cleaned.strip()

        # Determine handles for the output.
        # The comment section contains both the quote retweeter's handle
        # (at the top of the section) and the original author's handle
        # (on the embedded tweet card at the bottom of the section).
        # detect_handles() finds them in visual order, so:
        #   - The LAST handle in comment_handles is the original author
        #   - The FIRST handle in comment_handles is the quote retweeter
        # Fall back to original_handles if comment_handles is empty.
        #
        # IMPORTANT: When original_handles_are_mentions is True, the handles
        # in original_handles are just @mentions within the tweet content
        # (e.g. "#nw first watch w/ @SILNTKLL"), NOT the original author's
        # poster handle. In this case, the original author's handle is the
        # LAST handle from the comment section (which represents the embedded
        # tweet card's header). We should NOT use original_handles[0] as the
        # original author.
        if len(comment_handles) >= 2:
            # Both handles found in comment section
            handle_a = comment_handles[-1]  # Last = original author
            handle_b = comment_handles[0]   # First = quote retweeter
        elif len(comment_handles) == 1:
            # Only one handle found in comment section.
            # If original_handles are just content mentions (not poster handles),
            # the original author is the same as the comment section handle
            # (the embedded tweet card's author handle was merged into the
            # comment section by OCR). Otherwise, use original_handles[0].
            if original_handles_are_mentions:
                handle_a = comment_handles[0]  # Original author = same handle
                handle_b = comment_handles[0]  # Quote retweeter = same handle
            else:
                handle_b = comment_handles[0]
                handle_a = original_handles[0] if original_handles else "@original"
        else:
            # No handles in comment section
            handle_a = original_handles[0] if original_handles else "@original"
            handle_b = original_handles[-1] if len(original_handles) > 1 else (
                original_handles[0] if original_handles else "@commenter"
            )

        return (
            f"quote retweet. the original tweet is by {handle_a} and says\n\n"
            f"> {original_cleaned}\n\n"
            f"{handle_b} then quote retweets this and says\n"
            f"> {comment_cleaned}"
        )

    def format_as_reddit_post(self, text):
        """WIP: Return raw OCR text with timestamps stripped."""
        if not text:
            return text
        return self.strip_timestamps(text)

    def format_as_reddit_comment(self, text):
        """WIP: Return raw OCR text with timestamps stripped."""
        if not text:
            return text
        return self.strip_timestamps(text)

    def format_as_reddit_thread(self, text):
        """WIP: Return raw OCR text with timestamps stripped."""
        if not text:
            return text
        return self.strip_timestamps(text)

    def paste_from_clipboard(self, event=None):
        """Handle image paste from clipboard"""
        try:
            # Get image from clipboard (requires PIL)
            clipboard_image = ImageGrab.grabclipboard()
            
            if clipboard_image is None:
                self.status_var.set("No image found in clipboard. Copy an image first (Print Screen or Ctrl+C on an image)")
                return
                
            if isinstance(clipboard_image, Image.Image):
                self.current_image = clipboard_image
                self.annotated_image = None
                self.display_image(self.current_image)
                self.status_var.set("Image pasted successfully! Click 'Process Image' to extract text")
                self.process_button.config(state='normal')
                self.clear_button.config(state='normal')
                if self.auto_process_var.get():
                    self.root.after(100, self.process_image)
            else:
                self.status_var.set("Clipboard does not contain an image. Please copy an image first.")
                
        except Exception as e:
            self.status_var.set(f"Error pasting image: {str(e)}")
            messagebox.showerror("Error", f"Failed to paste image: {str(e)}")
    
    def display_image(self, image):
        """Display the image in the GUI with proper scaling"""
        # Get the image frame dimensions
        self.image_frame.update_idletasks()
        max_width = self.image_frame.winfo_width() - 20
        max_height = 400
        
        if max_width <= 0:
            max_width = 400
            
        # Calculate scaling factor
        img_width, img_height = image.size
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        new_size = (int(img_width * scale), int(img_height * scale))
        
        # Resize and display
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_image)
        
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
        
    def process_image(self):
        """Process the image with PaddleOCR to extract text"""
        if self.current_image is None or self.ocr_model is None:
            return
            
        try:
            self.status_var.set("Processing image with PaddleOCR...")
            self.root.update()
            
            # Resize large images to prevent memory exhaustion during OCR.
            # Very tall images (e.g. Twitter thread screenshots) can cause
            # PaddleOCR to detect hundreds of text blocks, each requiring a
            # separate recognition inference — which can exhaust system memory.
            # We limit the max dimension to 1280px while preserving aspect ratio.
            img = self.current_image
            max_dim = 1280
            scale_x = 1.0
            scale_y = 1.0
            if max(img.size) > max_dim:
                scale = max_dim / max(img.size)
                new_size = (int(img.width * scale), int(img.height * scale))
                scale_x = self.current_image.width / new_size[0]
                scale_y = self.current_image.height / new_size[1]
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.status_var.set(f"Resized image from {self.current_image.size} to {new_size} for OCR...")
                self.root.update()
            
            # Convert PIL Image to format PaddleOCR expects
            # Save to bytes, then read - simpler approach
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                img.save(temp_path, 'PNG')
            
            # Perform OCR using PaddleOCR.
            # Force the text detection model to internally resize images so the
            # longest side is at most 960px (limit_type="max"). The default config
            # uses limit_type="min" with limit_side_len=64, which acts as a minimum
            # size floor — it does NOT downscale large images. Without this, dense
            # screenshots like Twitter threads produce many text blocks that exhaust
            # memory during text recognition.
            result = self.ocr_model.predict(
                temp_path,
                text_det_limit_side_len=960,
                text_det_limit_type="max",
            )
        
            # Clean up temp file
            os.unlink(temp_path)
            
            if result and len(result) > 0 and result[0] is not None:
                # Extract text from PaddleOCR result structure
                min_confidence = self.confidence_var.get()
                extracted_lines = []
                confidence_info = []
                rec_boxes = []  # Store bounding boxes for paragraph detection
                
                annotated_image = self.current_image.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                # Helper to scale polygon coordinates from OCR-resized space back
                # to original image space, so red boxes are drawn correctly.
                def scale_poly(poly):
                    return [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in poly]
                
                # result[0] contains the detection results for the first image
                # Access .json once to avoid triggering _to_json() multiple times
                # (hasattr calls the property getter, so each access is expensive)
                result_json = result[0].json if hasattr(result[0], 'json') else None
                if result_json is not None and isinstance(result_json, dict) and 'res' in result_json:
                    res = result_json['res']
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    polys = res.get('dt_polys', []) or res.get('rec_polys', [])
                    
                    for i, (text, confidence) in enumerate(zip(texts, scores)):
                        if confidence >= min_confidence:
                            extracted_lines.append(text)
                            confidence_info.append(f"{confidence:.2f}")
                            if i < len(polys):
                                poly = polys[i]
                                # Scale polygon coordinates to original image space
                                scaled_poly = scale_poly(poly)
                                # Convert polygon to bounding box [x1, y1, x2, y2]
                                all_x = [p[0] for p in scaled_poly]
                                all_y = [p[1] for p in scaled_poly]
                                bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]
                                rec_boxes.append(bbox)
                                points = [(p[0], p[1]) for p in scaled_poly]
                                if points:
                                    points.append(points[0])
                                    draw.line(points, fill="red", width=2)
                elif isinstance(result[0], dict) and 'res' in result[0]:
                    res = result[0]['res']
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    polys = res.get('dt_polys', []) or res.get('rec_polys', [])
                    
                    for i, (text, confidence) in enumerate(zip(texts, scores)):
                        if confidence >= min_confidence:
                            extracted_lines.append(text)
                            confidence_info.append(f"{confidence:.2f}")
                            if i < len(polys):
                                poly = polys[i]
                                # Scale polygon coordinates to original image space
                                scaled_poly = scale_poly(poly)
                                # Convert polygon to bounding box [x1, y1, x2, y2]
                                all_x = [p[0] for p in scaled_poly]
                                all_y = [p[1] for p in scaled_poly]
                                bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]
                                rec_boxes.append(bbox)
                                points = [(p[0], p[1]) for p in scaled_poly]
                                if points:
                                    points.append(points[0])
                                    draw.line(points, fill="red", width=2)
                else:
                    for line in result[0]:
                        poly = line[0]
                        text = line[1][0]  # The recognized text
                        confidence = line[1][1]  # Confidence score (0-1)
                        
                        if confidence >= min_confidence:
                            extracted_lines.append(text)
                            confidence_info.append(f"{confidence:.2f}")
                            # Scale polygon coordinates to original image space
                            scaled_poly = scale_poly(poly)
                            # Convert polygon to bounding box
                            all_x = [p[0] for p in scaled_poly]
                            all_y = [p[1] for p in scaled_poly]
                            bbox = [min(all_x), min(all_y), max(all_x), max(all_y)]
                            rec_boxes.append(bbox)
                            points = [(p[0], p[1]) for p in scaled_poly]
                            if points:
                                points.append(points[0])
                                draw.line(points, fill="red", width=2)
                
                if extracted_lines:
                    # Save the raw OCR text as newline-joined extracted lines
                    self.raw_ocr_text = "\n".join(extracted_lines)
                    
                    # Create a cleaned version: strip statistics-only lines from raw text
                    # (pure noise like "90.3K Views"), then apply paragraph-aware formatting.
                    # IMPORTANT: Timestamps are PRESERVED in formatted_ocr_text so that
                    # formatters like format_as_quote_retweet and format_as_tweet_thread
                    # can use them for structural detection (e.g. finding the boundary
                    # between comment and original tweet). Each formatter strips timestamps
                    # itself as needed.
                    # Note: Lines like "8:44 · 04 Dec 23 · 90.3K Views" are caught by
                    # is_statistics_line() (Pattern 3) and removed here, which is correct
                    # since they are pure metadata with no tweet content.
                    # We need to filter rec_boxes to match the cleaned lines. Since
                    # strip_statistics removes entire lines, we track which original line
                    # indices survive by comparing the cleaned output lines back to the
                    # original lines.
                    original_lines = self.raw_ocr_text.split("\n")
                    
                    cleaned_raw = self.strip_statistics(self.raw_ocr_text)
                    cleaned_lines = cleaned_raw.split("\n")
                    
                    # Match cleaned lines back to original lines to filter rec_boxes.
                    # A cleaned line is either:
                    #   - an original line that was kept as-is
                    #   - an original line that was removed (stat line) — skip its rec_box
                    # We walk through both lists simultaneously to find matches.
                    cleaned_rec_boxes = []
                    orig_idx = 0
                    for cleaned_line in cleaned_lines:
                        # Skip original lines that were removed entirely
                        while orig_idx < len(original_lines):
                            orig_line = original_lines[orig_idx]
                            # Check if this original line matches the cleaned line
                            if cleaned_line == orig_line:
                                # Found the match — keep this rec_box
                                if orig_idx < len(rec_boxes):
                                    cleaned_rec_boxes.append(rec_boxes[orig_idx])
                                orig_idx += 1
                                break
                            # This original line was removed — skip its rec_box
                            orig_idx += 1
                    
                    self.formatted_ocr_text = self.format_text_with_paragraphs(cleaned_lines, cleaned_rec_boxes)
                    
                    # Apply selected output formatting using the cleaned + formatted text
                    output_type = self.output_type_var.get()
                    formatter = {
                        "tweet": self.format_as_tweet,
                        "tweet thread": self.format_as_tweet_thread,
                        "quote retweet": self.format_as_quote_retweet,
                        "reddit post": self.format_as_reddit_post,
                        "reddit comment": self.format_as_reddit_comment,
                        "reddit thread": self.format_as_reddit_thread,
                    }
                    formatter_func = formatter.get(output_type, self.format_as_tweet)
                    full_text = formatter_func(self.formatted_ocr_text)
                    
                    # Clear text widget and insert formatted text
                    self.text_widget.delete(1.0, tk.END)
                    self.text_widget.insert(1.0, full_text)
                    
                    self.annotated_image = annotated_image
                    self.display_image(self.annotated_image)
                    
                    # Optionally show confidence info in status
                    avg_conf = sum(float(c) for c in confidence_info) / len(confidence_info) if confidence_info else 0
                    self.status_var.set(f"OCR complete! Extracted {len(extracted_lines)} text blocks. Avg confidence: {avg_conf:.2f}")
                    self.copy_button.config(state='normal')
                else:
                    self.text_widget.delete(1.0, tk.END)
                    self.text_widget.insert(1.0, f"No text detected above confidence threshold ({min_confidence}).\n\nTry:\n• Lowering the confidence threshold\n• Using a clearer image\n• Selecting a different language")
                    self.status_var.set("No text detected above confidence threshold")
            else:
                self.text_widget.delete(1.0, tk.END)
                self.text_widget.insert(1.0, "No text detected in the image.\n\nTry:\n• Using a clearer image\n• Selecting a different language")
                self.status_var.set("No text detected in the image")
                
        except Exception as e:
            self.status_var.set(f"OCR Error: {str(e)}")
            messagebox.showerror("OCR Error", f"Failed to process image: {str(e)}")
    
    def on_output_type_change(self, event=None):
        """Re-format the displayed text when the output type dropdown changes."""
        if self.formatted_ocr_text is None:
            return  # No OCR results yet

        output_type = self.output_type_var.get()
        formatter = {
            "tweet": self.format_as_tweet,
            "tweet thread": self.format_as_tweet_thread,
            "quote retweet": self.format_as_quote_retweet,
            "reddit post": self.format_as_reddit_post,
            "reddit comment": self.format_as_reddit_comment,
            "reddit thread": self.format_as_reddit_thread,
        }
        formatter_func = formatter.get(output_type, self.format_as_tweet)
        full_text = formatter_func(self.formatted_ocr_text)

        # Update the text widget with the newly formatted text
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, full_text)
        self.status_var.set(f"Reformatted as {output_type}")

    def copy_to_clipboard(self):
        """Copy extracted text to clipboard with output formatting"""
        text = self.text_widget.get(1.0, tk.END).strip()
        if text:
            output_type = self.output_type_var.get()
            formatter = {
                "tweet": self.format_as_tweet,
                "tweet thread": self.format_as_tweet_thread,
                "quote retweet": self.format_as_quote_retweet,
                "reddit post": self.format_as_reddit_post,
                "reddit comment": self.format_as_reddit_comment,
                "reddit thread": self.format_as_reddit_thread,
            }
            formatter_func = formatter.get(output_type, self.format_as_tweet)
            # Use formatted_ocr_text if available, otherwise fall back to widget text
            source_text = self.formatted_ocr_text if self.formatted_ocr_text else text
            formatted_text = formatter_func(source_text)
            pyperclip.copy(formatted_text)
            self.status_var.set(
                f"Copied {len(formatted_text)} characters as {output_type}!"
            )

            # Flash the copy button to provide visual feedback
            self.copy_button.config(text="✓ Copied!")
            self.root.after(2000, lambda: self.copy_button.config(text="📋 Copy to Clipboard"))
    
    def clear_all(self):
        """Clear the image and text"""
        self.current_image = None
        self.annotated_image = None
        self.current_ocr_result = None
        self.raw_ocr_text = None
        self.formatted_ocr_text = None
        self.image_label.config(image='', text="No image pasted yet\n\nPress Ctrl+V to paste an image")
        self.image_label.image = None
        self.text_widget.delete(1.0, tk.END)
        self.copy_button.config(state='disabled')
        self.process_button.config(state='disabled')
        self.status_var.set("Cleared - Press Ctrl+V to paste a new image")


def main():
    root = TkinterDnD.Tk()
    app = PaddleOCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()