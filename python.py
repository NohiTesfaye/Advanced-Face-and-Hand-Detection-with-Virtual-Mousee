import warnings
import mediapipe as mp
import cv2
import numpy as np
import time
from datetime import datetime
import math
import pyautogui
from pynput.mouse import Button, Controller
import os

# Suppress the specific protobuf warning
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype.*")

class AdvancedFaceHandDetector:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize models with higher accuracy
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mouse control variables
        self.mouse = Controller()
        self.screen_width, self.screen_height = pyautogui.size()
        self.virtual_mouse_enabled = False
        self.mouse_smoothing = []
        self.smoothing_buffer_size = 5
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds
        self.cursor_visible = True
        self.cursor_blink_time = time.time()
        self.is_dragging = False
        
        # Scroll functionality
        self.scroll_enabled = False
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.3
        
        # Tracking variables
        self.face_counter = 0
        self.hand_counter = 0
        self.start_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Gesture recognition
        self.gesture_history = []
        self.last_gesture_time = 0
        
        # UI settings
        self.show_landmarks = True
        self.show_bounding_boxes = True
        self.show_metrics = True
        self.show_gestures = True
        self.mirror_view = True
        
        # Performance tracking
        self.performance_log = []
        self.last_log_time = time.time()

    def take_screenshot(self, image):
        """Take screenshot and save with timestamp"""
        # Create screenshots directory if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"screenshots/screenshot_{timestamp}.png"
        
        # Save the image
        cv2.imwrite(filename, image)
        print(f"📸 Screenshot saved: {filename}")
        
        # Show confirmation message on screen
        cv2.putText(image, "SCREENSHOT SAVED!", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        return filename

    def take_annotated_screenshot(self, image, hand_count=0, face_count=0):
        """Take screenshot with annotations and metadata"""
        # Create screenshots directory if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        
        # Create a copy to avoid modifying original
        screenshot = image.copy()
        
        # Add metadata overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_text = [
            f"Timestamp: {timestamp}",
            f"Hands Detected: {hand_count}",
            f"Faces Detected: {face_count}",
            f"FPS: {self.fps:.1f}",
            f"Virtual Mouse: {'ON' if self.virtual_mouse_enabled else 'OFF'}"
        ]
        
        # Add metadata to screenshot
        for i, text in enumerate(metadata_text):
            cv2.putText(screenshot, text, (10, image.shape[0] - 50 - i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add watermark
        cv2.putText(screenshot, "AI Vision System", (image.shape[1] - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Generate filename
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"screenshots/screenshot_{timestamp_file}.png"
        
        # Save the image
        cv2.imwrite(filename, screenshot)
        print(f"📸 Screenshot saved with metadata: {filename}")
        
        return filename

    def enable_virtual_mouse(self):
        """Enable virtual mouse control"""
        self.virtual_mouse_enabled = True
        print("🖱️ Virtual Mouse ENABLED - Use index finger to control cursor")

    def disable_virtual_mouse(self):
        """Disable virtual mouse control"""
        self.virtual_mouse_enabled = False
        self.is_dragging = False
        print("🖱️ Virtual Mouse DISABLED")

    def toggle_scroll_mode(self):
        """Toggle between cursor and scroll modes"""
        self.scroll_enabled = not self.scroll_enabled
        mode = "SCROLL MODE 🔄" if self.scroll_enabled else "CURSOR MODE 🖱️"
        print(f"📜 {mode}")

    def smooth_mouse_movement(self, x, y):
        """Apply smoothing to mouse movements"""
        self.mouse_smoothing.append((x, y))
        if len(self.mouse_smoothing) > self.smoothing_buffer_size:
            self.mouse_smoothing.pop(0)
        
        # Calculate average
        avg_x = sum([pos[0] for pos in self.mouse_smoothing]) / len(self.mouse_smoothing)
        avg_y = sum([pos[1] for pos in self.mouse_smoothing]) / len(self.mouse_smoothing)
        
        return int(avg_x), int(avg_y)

    def control_virtual_mouse(self, hand_landmarks, image_shape):
        """Control mouse using hand landmarks"""
        if not self.virtual_mouse_enabled:
            return
        
        h, w = image_shape[:2]
        
        # Get finger positions
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        pinky_tip = hand_landmarks.landmark[20]
        
        if self.scroll_enabled:
            self.handle_scroll_gesture(hand_landmarks, image_shape)
            return
        
        # Convert to screen coordinates for cursor mode
        screen_x = int(index_tip.x * self.screen_width)
        screen_y = int(index_tip.y * self.screen_height)
        
        # Apply smoothing
        smooth_x, smooth_y = self.smooth_mouse_movement(screen_x, screen_y)
        
        # Move mouse
        try:
            self.mouse.position = (smooth_x, smooth_y)
        except Exception as e:
            print(f"Mouse control error: {e}")
        
        # Check for click gestures
        self.handle_click_gestures(hand_landmarks, image_shape)
        
        # Check for drag gesture
        self.handle_drag_gesture(hand_landmarks, image_shape)

    def handle_scroll_gesture(self, hand_landmarks, image_shape):
        """Handle scroll gestures"""
        h, w = image_shape[:2]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        
        # Calculate vertical position for scroll
        scroll_y = middle_tip.y * h
        
        # Scroll up/down based on hand position
        current_time = time.time()
        if current_time - self.last_scroll_time > self.scroll_cooldown:
            if scroll_y < h * 0.4:  # Top of screen
                self.mouse.scroll(0, 2)  # Scroll up
                print("📜 SCROLL UP")
                self.last_scroll_time = current_time
            elif scroll_y > h * 0.6:  # Bottom of screen
                self.mouse.scroll(0, -2)  # Scroll down
                print("📜 SCROLL DOWN")
                self.last_scroll_time = current_time

    def handle_click_gestures(self, hand_landmarks, image_shape):
        """Handle mouse click gestures"""
        h, w = image_shape[:2]
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        thumb_index_dist = self.calculate_distance(
            (thumb_tip.x * w, thumb_tip.y * h),
            (index_tip.x * w, index_tip.y * h)
        )
        
        thumb_middle_dist = self.calculate_distance(
            (thumb_tip.x * w, thumb_tip.y * h),
            (middle_tip.x * w, middle_tip.y * h)
        )
        
        current_time = time.time()
        
        # Left click (thumb-index pinch)
        if thumb_index_dist < 30 and (current_time - self.last_click_time) > self.click_cooldown:
            self.mouse.click(Button.left, 1)
            self.last_click_time = current_time
            print("🖱️ LEFT CLICK")
        
        # Right click (thumb-middle pinch)
        if thumb_middle_dist < 30 and (current_time - self.last_click_time) > self.click_cooldown:
            self.mouse.click(Button.right, 1)
            self.last_click_time = current_time
            print("🖱️ RIGHT CLICK")

    def handle_drag_gesture(self, hand_landmarks, image_shape):
        """Handle drag and drop gestures"""
        h, w = image_shape[:2]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Check if fingers are close together for drag
        finger_dist = self.calculate_distance(
            (index_tip.x * w, index_tip.y * h),
            (middle_tip.x * w, middle_tip.y * h)
        )
        
        if finger_dist < 40:  # Fingers close together
            if not self.is_dragging:
                self.mouse.press(Button.left)
                self.is_dragging = True
                print("🖱️ DRAG START")
        elif self.is_dragging:
            self.mouse.release(Button.left)
            self.is_dragging = False
            print("🖱️ DRAG END")

    def draw_virtual_mouse_ui(self, image, hand_landmarks, image_shape):
        """Draw virtual mouse UI elements"""
        if not self.virtual_mouse_enabled or not hand_landmarks:
            return
        
        h, w = image.shape[:2]
        
        # Get finger positions
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        # Convert to image coordinates
        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)
        middle_x = int(middle_tip.x * w)
        middle_y = int(middle_tip.y * h)
        
        # Draw cursor based on mode
        if self.scroll_enabled:
            cursor_color = (255, 0, 255)  # Purple for scroll mode
            mode_text = "SCROLL MODE"
        else:
            cursor_color = (0, 255, 255)  # Yellow for cursor mode
            mode_text = "CURSOR MODE"
        
        cursor_size = 8
        
        # Blinking cursor effect
        if time.time() - self.cursor_blink_time > 0.5:
            self.cursor_visible = not self.cursor_visible
            self.cursor_blink_time = time.time()
        
        if self.cursor_visible:
            # Draw crosshair cursor
            cv2.circle(image, (index_x, index_y), cursor_size, cursor_color, 2)
            cv2.line(image, (index_x - cursor_size, index_y), 
                    (index_x + cursor_size, index_y), cursor_color, 2)
            cv2.line(image, (index_x, index_y - cursor_size), 
                    (index_x, index_y + cursor_size), cursor_color, 2)
            
            # Draw mode text
            cv2.putText(image, mode_text, (index_x + 20, index_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cursor_color, 2)
        
        # Draw click indicators
        thumb_index_dist = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
        thumb_middle_dist = self.calculate_distance((thumb_x, thumb_y), (middle_x, middle_y))
        
        # Left click indicator (thumb-index)
        if thumb_index_dist < 30:
            cv2.putText(image, "LEFT CLICK", (index_x + 15, index_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)
        
        # Right click indicator (thumb-middle)
        if thumb_middle_dist < 30:
            cv2.putText(image, "RIGHT CLICK", (index_x + 15, index_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.line(image, (thumb_x, thumb_y), (middle_x, middle_y), (0, 0, 255), 3)

    def calculate_fps(self):
        """Calculate frames per second"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = current_time
            
            # Log performance every 5 seconds
            if current_time - self.last_log_time >= 5:
                self.performance_log.append(self.fps)
                if len(self.performance_log) > 10:
                    self.performance_log.pop(0)
                self.last_log_time = current_time

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def detect_gestures(self, hand_landmarks, image_shape):
        """Detect hand gestures"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * image_shape[1])
            y = int(landmark.y * image_shape[0])
            landmarks.append((x, y))
        
        # Thumb and index finger positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        fingers_up = 0
        
        # Check which fingers are extended
        if index_tip[1] < landmarks[6][1]:  # Index finger up
            fingers_up += 1
        if middle_tip[1] < landmarks[10][1]:  # Middle finger up
            fingers_up += 1
        if ring_tip[1] < landmarks[14][1]:  # Ring finger up
            fingers_up += 1
        if pinky_tip[1] < landmarks[18][1]:  # Pinky up
            fingers_up += 1
        if thumb_tip[0] > landmarks[3][0]:  # Thumb up (right hand)
            fingers_up += 1
        
        # Gesture recognition
        gesture = "Unknown"
        
        if thumb_index_dist < 30:
            gesture = "Pinch 👌"
        elif fingers_up == 5:
            gesture = "Open Hand ✋"
        elif fingers_up == 0:
            gesture = "Fist ✊"
        elif fingers_up == 1 and index_tip[1] < landmarks[6][1]:
            gesture = "Pointing 👆"
        elif fingers_up == 2 and index_tip[1] < landmarks[6][1] and middle_tip[1] < landmarks[10][1]:
            gesture = "Victory ✌️"
        elif fingers_up == 3 and thumb_tip[0] > landmarks[3][0]:
            gesture = "OK 👌"
        
        return gesture, fingers_up

    def draw_enhanced_landmarks(self, image, hand_results, face_results, face_mesh_results):
        """Draw enhanced landmarks and annotations"""
        h, w = image.shape[:2]
        
        # Draw hand landmarks and gestures
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if self.show_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Virtual mouse control (use first hand only)
                if i == 0 and self.virtual_mouse_enabled:
                    self.control_virtual_mouse(hand_landmarks, image.shape)
                    self.draw_virtual_mouse_ui(image, hand_landmarks, image.shape)
                
                # Get gesture
                gesture, fingers_count = self.detect_gestures(hand_landmarks, image.shape)
                
                # Draw hand bounding box
                if self.show_bounding_boxes:
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, f'Hand {i+1}', (x_min, y_min-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display gesture
                if self.show_gestures:
                    cv2.putText(image, f'Gesture: {gesture}', (10, 60 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw face detections
        if face_results.detections:
            for i, detection in enumerate(face_results.detections):
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                if self.show_bounding_boxes:
                    cv2.rectangle(image, (x_min, y_min), (x_min+width, y_min+height), (255, 0, 0), 2)
                    cv2.putText(image, f'Face {i+1}', (x_min, y_min-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw face mesh
        if face_mesh_results.multi_face_landmarks and self.show_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        
        return image

    def draw_metrics_panel(self, image, hand_count, face_count):
        """Draw metrics and information panel"""
        h, w = image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Virtual mouse status
        mouse_status = "ENABLED 🖱️" if self.virtual_mouse_enabled else "DISABLED"
        status_color = (0, 255, 0) if self.virtual_mouse_enabled else (0, 0, 255)
        
        scroll_status = "SCROLL 🔄" if self.scroll_enabled else "CURSOR 🖱️"
        scroll_color = (255, 0, 255) if self.scroll_enabled else (0, 255, 255)
        
        # Display metrics
        if self.show_metrics:
            y_offset = 20
            line_height = 20
            
            metrics = [
                f'FPS: {self.fps:.1f}',
                f'Hands: {hand_count}',
                f'Faces: {face_count}',
                f'Mouse: {mouse_status}',
                f'Mode: {scroll_status}',
                f'Time: {datetime.now().strftime("%H:%M:%S")}',
                f'Frame: {self.frame_count}',
                f'Drag: {"ON" if self.is_dragging else "OFF"}',
                f'Screenshot: Press P'
            ]
            
            colors = [
                (0, 255, 255),  # FPS - Yellow
                (0, 255, 0),    # Hands - Green
                (255, 0, 0),    # Faces - Blue
                status_color,    # Mouse status
                scroll_color,    # Mode
                (255, 255, 255), # Time - White
                (255, 255, 255), # Frame - White
                (255, 165, 0),  # Drag - Orange
                (255, 255, 0)   # Screenshot - Yellow
            ]
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                cv2.putText(image, metric, (10, y_offset + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw instructions
        instructions = [
            "H - Toggle landmarks",
            "B - Bounding boxes", 
            "M - Metrics display",
            "V - Virtual Mouse",
            "S - Scroll Mode",
            "P - Screenshot",
            "Q - Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, (w-300, 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        if self.mirror_view:
            frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process all models
        hand_results = self.hands.process(image_rgb)
        face_results = self.face_detection.process(image_rgb)
        face_mesh_results = self.face_mesh.process(image_rgb)
        
        # Convert back to BGR for drawing
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw enhanced visualizations
        self.draw_enhanced_landmarks(image_bgr, hand_results, face_results, face_mesh_results)
        
        # Update counters
        hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        face_count = len(face_results.detections) if face_results.detections else 0
        
        # Draw metrics panel
        self.draw_metrics_panel(image_bgr, hand_count, face_count)
        
        # Calculate FPS
        self.calculate_fps()
        
        return image_bgr, hand_count, face_count

def main():
    detector = AdvancedFaceHandDetector()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("🚀 Advanced Face & Hand Detection with Virtual Mouse Started!")
    print("=== Controls ===")
    print("H - Toggle landmarks")
    print("B - Toggle bounding boxes")
    print("M - Toggle metrics")
    print("V - Toggle Virtual Mouse")
    print("S - Toggle Scroll Mode")
    print("P - Take Screenshot")
    print("Q - Quit")
    print("\n🖱️ Virtual Mouse Instructions:")
    print("- Point with index finger to move cursor")
    print("- Pinch thumb & index for LEFT CLICK")
    print("- Pinch thumb & middle for RIGHT CLICK")
    print("- Keep fingers together for DRAG")
    print("- In scroll mode: move hand up/down to scroll")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Process frame
        processed_frame, hand_count, face_count = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Advanced Face & Hand Detection + Virtual Mouse', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            detector.show_landmarks = not detector.show_landmarks
        elif key == ord('b'):
            detector.show_bounding_boxes = not detector.show_bounding_boxes
        elif key == ord('m'):
            detector.show_metrics = not detector.show_metrics
        elif key == ord('v'):
            if detector.virtual_mouse_enabled:
                detector.disable_virtual_mouse()
            else:
                detector.enable_virtual_mouse()
        elif key == ord('s'):
            if detector.virtual_mouse_enabled:
                detector.toggle_scroll_mode()
        elif key == ord('p'):  # Press 'P' for screenshot
            screenshot_path = detector.take_annotated_screenshot(
                processed_frame, 
                hand_count=hand_count, 
                face_count=face_count
            )
            # Visual feedback
            original_frame = processed_frame.copy()
            flash_frame = processed_frame.copy()
            flash_frame[:,:] = (0, 255, 255)  # Yellow flash
            cv2.imshow('Advanced Face & Hand Detection + Virtual Mouse', flash_frame)
            cv2.waitKey(50)  # Short flash
            cv2.imshow('Advanced Face & Hand Detection + Virtual Mouse', original_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application closed successfully!")
    
    # Print performance summary
    if detector.performance_log:
        avg_fps = sum(detector.performance_log) / len(detector.performance_log)
        print(f"📊 Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    main()
