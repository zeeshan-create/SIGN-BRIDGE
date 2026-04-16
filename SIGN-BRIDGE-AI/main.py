"""
SignBridge AI - Unified Python Backend
======================================
Combines Node.js backend + Python bridge into single Flask+SocketIO server.
Port: 5000 (same as original Node.js backend)
"""

import os
import sys
import json
import time
import base64
import bcrypt
import jwt
import threading
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from werkzeug.utils import secure_filename
import cv2

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
UPLOAD_DIR = BASE_DIR / "uploads"
USERS_FILE = BASE_DIR / "users.json"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Import ASL ML Engine ---
try:
    from real_time_detection import ASLInterpreter, CameraStream
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Failed to import ASL modules from {SRC_DIR}: {e}")
    sys.exit(1)

# --- Flask & SocketIO Setup ---
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Use threading mode for compatibility with Windows and eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# --- Configuration ---
JWT_SECRET = os.environ.get('JWT_SECRET', 'signbridge_elite_secret_2026')
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
JWT_EXPIRY_DAYS = 7

# --- Global State ---
interpreter = ASLInterpreter()
camera = None
camera_active = False
lock = threading.Lock()
session_history = []
current_sentence_id = 0

# --- User Data Management ---
def get_users():
    """Load users from JSON file."""
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)

# --- JWT Utilities ---
def generate_token(user_id, role):
    """Generate JWT token compatible with frontend."""
    payload = {
        'id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_token(token):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to protect routes with JWT."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                pass
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        payload = verify_token(token)
        if not payload:
            return jsonify({'message': 'Token is invalid or expired'}), 401
        
        return f(payload, *args, **kwargs)
    return decorated

# --- Auth Routes ---
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user with bcrypt password hashing."""
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if not all([name, email, password]):
        return jsonify({'message': 'Missing required fields'}), 400
    
    users = get_users()
    
    # Check if user already exists
    if any(u['email'] == email for u in users):
        return jsonify({'message': 'User already exists'}), 400
    
    # Hash password with bcrypt (compatible with Node.js bcryptjs)
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(10)).decode('utf-8')
    
    # First user is admin
    role = 'admin' if len(users) == 0 else 'user'
    
    new_user = {
        'id': int(time.time() * 1000),  # Same format as Node.js Date.now()
        'name': name,
        'email': email,
        'password': hashed_password,
        'role': role
    }
    
    users.append(new_user)
    save_users(users)
    
    # Generate token
    token = generate_token(new_user['id'], role)
    
    return jsonify({
        'token': token,
        'user': {
            'id': new_user['id'],
            'name': name,
            'email': email,
            'role': role
        }
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user with bcrypt password verification."""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({'message': 'Missing email or password'}), 400
    
    users = get_users()
    user = next((u for u in users if u['email'] == email), None)
    
    if not user:
        return jsonify({'message': 'User not found'}), 400
    
    # Check if user has password (Google-only users might not)
    if 'password' not in user:
        return jsonify({'message': 'Please use Google login'}), 400
    
    # Verify password with bcrypt
    stored_hash = user['password']
    # Handle both $2b$ and $2a$ prefixes (Node.js uses $2a$, Python uses $2b$)
    if stored_hash.startswith('$2a$'):
        stored_hash = '$2b$' + stored_hash[4:]
    
    if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
        return jsonify({'message': 'Invalid credentials'}), 400
    
    # Generate token
    token = generate_token(user['id'], user['role'])
    
    return jsonify({
        'token': token,
        'user': {
            'id': user['id'],
            'name': user['name'],
            'email': user['email'],
            'role': user['role'],
            'image': user.get('image')
        }
    })

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Verify Google OAuth token and create/login user."""
    data = request.get_json()
    google_token = data.get('token')
    
    if not google_token:
        return jsonify({'message': 'Google token is required'}), 400
    
    try:
        # Verify Google token
        idinfo = id_token.verify_oauth2_token(
            google_token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )
        
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Invalid issuer')
        
        email = idinfo.get('email')
        name = idinfo.get('name')
        google_id = idinfo.get('sub')
        picture = idinfo.get('picture')
        
        users = get_users()
        user = next((u for u in users if u['email'] == email), None)
        
        if not user:
            # Create new user from Google data
            role = 'admin' if len(users) == 0 else 'user'
            user = {
                'id': int(time.time() * 1000),
                'name': name,
                'email': email,
                'googleId': google_id,
                'role': role,
                'image': picture
            }
            users.append(user)
            save_users(users)
        else:
            # Update existing user with Google data if needed
            if not user.get('googleId'):
                user['googleId'] = google_id
                user['image'] = picture
                save_users(users)
        
        # Generate token
        token = generate_token(user['id'], user['role'])
        
        return jsonify({
            'token': token,
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'role': user['role'],
                'image': user.get('image')
            }
        })
        
    except ValueError as e:
        print(f"[Auth] Google token verification failed: {e}")
        return jsonify({'message': 'Invalid Google token'}), 401
    except Exception as e:
        print(f"[Auth] Google auth error: {e}")
        return jsonify({'message': 'Google authentication failed'}), 500

# --- Video Upload Route ---
@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Handle video file uploads."""
    if 'video' not in request.files:
        return jsonify({'message': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    # Get start frame from form data
    start_frame = request.form.get('startFrame', '0')
    try:
        start_frame = int(start_frame)
    except ValueError:
        start_frame = 0
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}_{filename}"
    filepath = UPLOAD_DIR / filename
    
    file.save(filepath)
    
    print(f"[Server] Video uploaded: {filepath} (startFrame: {start_frame})")
    
    # Process video asynchronously
    threading.Thread(target=process_video_file, args=(str(filepath), start_frame), daemon=True).start()
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'filename': filename,
        'path': str(filepath),
        'size': os.path.getsize(filepath),
        'startFrame': start_frame
    })

def process_video_file(video_path, start_frame=0):
    """Process uploaded video file and emit predictions via SocketIO."""
    global camera_active
    
    print(f"[Server] Processing video: {video_path} from frame {start_frame}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Server] Error: Could not open video {video_path}")
            return
        
        # Seek to start frame if specified
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame through ASL interpreter
            frame = cv2.flip(frame, 1)
            processed_frame = interpreter.process_frame(frame)
            
            # Encode for transfer
            _, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            
            current_sign = interpreter.history[-1] if interpreter.history else "---"
            current_sentence = " ".join(interpreter.sentence)
            
            # Emit to all connected clients
            socketio.emit('prediction', {
                "image": frame_base64,
                "sign": current_sign,
                "sentence": current_sentence,
                "fps": round(interpreter._calculate_fps(), 1),
                "mode": interpreter.mode
            })
            
            frame_count += 1
            
            # Small delay to not overwhelm the client
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        print(f"[Server] Video processing complete: {frame_count} frames processed")
        
        # Update history
        if interpreter.sentence:
            sentence_text = " ".join(interpreter.sentence)
            global session_history, current_sentence_id
            
            current_log = next((h for h in session_history if h['id'] == current_sentence_id), None)
            if current_log:
                if current_log['sentence'] != sentence_text:
                    current_log['sentence'] = sentence_text
                    socketio.emit('history_updated', session_history)
            else:
                session_history.append({'id': current_sentence_id, 'sentence': sentence_text})
                socketio.emit('history_updated', session_history)
        
    except Exception as e:
        print(f"[Server] Video processing error: {e}")

# --- ML Detection Loop ---
def detection_loop():
    """Main camera detection loop."""
    global camera, camera_active, interpreter
    
    print("[Python Bridge] AI Loop entering frame cycle...")
    
    while camera_active:
        if camera is None:
            break
        
        try:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            processed_frame = interpreter.process_frame(frame)
            
            # Encode for transfer
            _, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            
            current_sign = interpreter.history[-1] if interpreter.history else "---"
            current_sentence = " ".join(interpreter.sentence)
            
            # Emit results
            socketio.emit('prediction', {
                "image": frame_base64,
                "sign": current_sign,
                "sentence": current_sentence,
                "fps": round(interpreter._calculate_fps(), 1),
                "mode": interpreter.mode
            })
            
            # Update session history
            global session_history, current_sentence_id
            if interpreter.sentence:
                sentence_text = " ".join(interpreter.sentence)
                current_log = next((h for h in session_history if h['id'] == current_sentence_id), None)
                if current_log:
                    if current_log['sentence'] != sentence_text:
                        current_log['sentence'] = sentence_text
                        socketio.emit('history_updated', session_history)
                else:
                    session_history.append({'id': current_sentence_id, 'sentence': sentence_text})
                    socketio.emit('history_updated', session_history)
            
        except Exception as e:
            print(f"[Python Bridge] Loop Crash Avoided: {e}")
        
        time.sleep(0.01)  # Power safety

# --- Socket.IO Events ---
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"[Server] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"[Server] Client disconnected: {request.sid}")

@socketio.on('start_detection')
def handle_start_detection():
    """Start camera stream and AI detection."""
    global camera, camera_active
    
    print("[Server] Received start_detection")
    
    with lock:
        if not camera_active:
            print("[Python Bridge] Initializing camera hardware...")
            try:
                camera = CameraStream(0)
                if not camera.is_opened:
                    print("[Python Bridge] Camera 0 failed, trying index 1...")
                    camera = CameraStream(1)
                
                if not camera.is_opened:
                    emit('stream_status', {'status': 'error', 'message': 'No hardware camera detected'})
                    return
                
                camera_active = True
                # Start AI Loop in background
                t = threading.Thread(target=detection_loop, name="AI_Worker", daemon=True)
                t.start()
                print("[Python Bridge] AI Worker started.")
                emit('stream_status', {'status': 'active'})
            except Exception as e:
                print(f"[Python Bridge] Camera Error: {e}")
                camera_active = False
                emit('stream_status', {'status': 'error', 'message': str(e)})
        else:
            emit('stream_status', {'status': 'already_active'})

@socketio.on('stop_detection')
def handle_stop_detection():
    """Stop camera stream and AI detection."""
    global camera, camera_active
    
    print("[Server] Received stop_detection")
    
    with lock:
        camera_active = False
        if camera:
            camera.release()
        camera = None
    
    print("[Python Bridge] Stream stopped.")
    emit('stream_status', {'status': 'stopped'})

@socketio.on('set_mode')
def handle_set_mode(mode):
    """Set ASL detection mode (ALPHABET, WORDS, ALL)."""
    global interpreter
    
    if isinstance(mode, dict):
        mode = mode.get('mode', 'ALL')
    
    mode = str(mode).upper() if mode else 'ALL'
    
    if mode in interpreter.MODES:
        interpreter.mode = mode
        interpreter.mode_idx = interpreter.MODES.index(mode)
        interpreter.history.clear()
        print(f"[Python Bridge] Mode -> {mode}")
        emit('mode_updated', {'mode': mode})
    else:
        print(f"[Python Bridge] Invalid mode: {mode}")

@socketio.on('clear_sentence')
def handle_clear_sentence():
    """Clear current sentence and history."""
    global current_sentence_id
    
    interpreter.sentence.clear()
    interpreter.history.clear()
    current_sentence_id += 1  # Move new sentences to new line in history
    print("[Python Bridge] Sentence cleared.")

@socketio.on('clear_history')
def handle_clear_history():
    """Clear entire session history."""
    global session_history, current_sentence_id
    
    session_history = []
    current_sentence_id += 1
    emit('history_updated', session_history)
    print("[Server] History cleared.")

@socketio.on('get_history')
def handle_get_history():
    """Return current session history."""
    emit('history_updated', session_history)

@socketio.on('manual_input')
def handle_manual_input(character):
    """Add manual character input to sentence."""
    if character and len(character) > 0:
        interpreter.sentence.append(character)
        print(f"[Python Bridge] Manual input added: {character}")

# --- Health Check ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': int(time.time() * 1000),
        'camera_active': camera_active
    })

# --- Production Initialization ---
def initialize_app():
    """Initialize app for production. Called when imported by gunicorn."""
    print("=" * 60)
    print("SignBridge AI - Unified Python Backend")
    print("=" * 60)
    print(f"[Server] Users file: {USERS_FILE}")
    print(f"[Server] Upload directory: {UPLOAD_DIR}")
    print(f"[Server] Environment: {'Production' if os.environ.get('RAILWAY_ENVIRONMENT') else 'Development'}")
    print("=" * 60)
    
    # Check if users file exists
    if not USERS_FILE.exists():
        print(f"[Warning] Users file not found at {USERS_FILE}")
        print("[Warning] Creating empty users file...")
        save_users([])
    else:
        users = get_users()
        print(f"[Server] Loaded {len(users)} users")
    
    return app

# Initialize when imported (for gunicorn)
app = initialize_app()

# --- Main Entry Point (Development Only) ---
if __name__ == "__main__":
    print(f"[Server] Starting development server on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
