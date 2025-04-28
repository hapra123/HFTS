import os
import uuid
import time
import pyodbc
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime
from scipy.spatial.distance import cosine
from flask import Flask, request, render_template, redirect, url_for, Response
from azure.storage.blob import BlobServiceClient
from deepface import DeepFace
import cv2
import ast
import logging
from geopy.geocoders import Nominatim
from collections import defaultdict
import time
import uuid
import cv2
from datetime import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Check if handlers are already set up
if not logger.handlers:
    # If no handlers, add a StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    
    # Format logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(stream_handler)
    print("Log handler added.")
# Store current location in a global variable or session
current_location = ""
# Azure Blob config
connection_str = "Putyourconnectionstringhere"
container_name = "faceframes"
blob_service_client = BlobServiceClient.from_connection_string(connection_str)
container_client = blob_service_client.get_container_client(container_name)
# Store recently uploaded faces (for 30 seconds check)
recently_uploaded = {}

# Define a delay of 30 seconds
UPLOAD_DELAY = 30

# Start video capture
camera = cv2.VideoCapture(0)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Flag to manage the 1-second delay between face detections
last_detection_time = time.time()

def get_db_connection():
    server = ''
    database = ''
    username = ''
    password = ''
    driver = '{ODBC Driver 18 for SQL Server}'

    conn_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    return pyodbc.connect(conn_str)

def get_face_embedding(image):
    embeddings = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
    if isinstance(embeddings, list):
        return embeddings[0]["embedding"]
    return np.array(embeddings).tolist()

def find_matching_user(face_embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT cust_id,name, face_vector FROM registered_users")
    rows = cursor.fetchall()
    conn.close()

    min_distance = float('inf')
    matched_user = None

    for cust_id, name, stored_vector_str in rows:
        try:
            stored_obj = ast.literal_eval(stored_vector_str)  # e.g. [{'embedding': [...]}]
            if isinstance(stored_obj, list) and 'embedding' in stored_obj[0]:
                stored_embedding = np.array(stored_obj[0]['embedding'])
                distance = cosine(face_embedding, stored_embedding)
                if distance < 0.5 and distance < min_distance:
                    min_distance = distance
                    matched_user = name
                    cust_id=cust_id
        except Exception as e:
            print(f"Error parsing embedding for {name}: {e}")

    return matched_user


# Create and configure engine once
  # Lock to avoid concurrent runAndWait



# Face tracking and caching
import cv2
import time
import uuid
import numpy as np
from collections import defaultdict
from datetime import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import simpleaudio as sa
import simpleaudio as sa

def play_sound(path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # This blocks until playback is finished
    except Exception as e:
        print(f"Audio play failed: {e}")
# --- Global State ---
last_upload_time = defaultdict(lambda: 0)
last_seen_faces = {}
embedding_cache = {}
spoken_users = set()

# --- Constants ---
UNKNOWN_DELAY = 50
REGISTERED_DELAY = 200
FACE_RETENTION_SECONDS = 5
SIMILARITY_THRESHOLD = 0.75

# Update with your audio path
BEEP_AUDIO_PATH = "beep-329314 (mp3cut.net).wav"
BEEP_AUDIO_PATH2="mixkit-retro-game-notification-212 (mp3cut.net).wav"

from datetime import datetime
import logging

# Assuming the logger is defined elsewhere in your code
logger = logging.getLogger(__name__)

from datetime import datetime

def deactivate_trip_for_registered_user(label):
    """
    Deactivates the trip for a registered user.
    Updates the trip to 'Done', sets exit_location to 'Mangalore', and records the end_time.
    Also deducts ₹50 from the user's balance.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get current timestamp for end_time
        current_time = datetime.now()

        # Fetch the cust_id and balance for the user
        cursor.execute("SELECT cust_id, balance FROM registered_users WHERE name = ?", (label,))
        result = cursor.fetchone()

        if result:
            cust_id, balance = result
            new_balance = balance - 50

            # Update the trip to mark it done and add exit info
            cursor.execute("""
                UPDATE trips
                SET 
                    is_active = 'Done',
                    exit_location = 'Mangalore',
                    end_time = ?
                WHERE cust_id = ? AND is_active = 'Active'
            """, (current_time, cust_id))

            # Update the user's balance
            cursor.execute("""
                UPDATE registered_users
                SET balance = ?
                WHERE cust_id = ?
            """, (new_balance, cust_id))

            conn.commit()
            print(f"Trip for user {label} successfully marked as 'Done'. Exit at Mangalore. ₹50 deducted. New balance: ₹{new_balance}.")
        else:
            print(f"User {label} not found in registered users.")
    
    except Exception as e:
        logger.error(f"Error deactivating trip and updating balance for user {label}: {e}")
    finally:
        conn.close()





def detect_faces_and_stream(camera, container_client, logger, get_face_embedding, find_matching_user, current_location):
    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()

        try:
            results = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv', enforce_detection=False)
            face_list = results if isinstance(results, list) else [results]

            for face in face_list:
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                if w == 0 or h == 0:
                    continue

                face_crop = frame[y:y + h, x:x + w]
                current_embedding = get_face_embedding(face_crop)

                # Match label
                label = "Unknown"
                for cached_label, cached_emb in embedding_cache.items():
                    sim = cosine_similarity([current_embedding], [cached_emb])[0][0]
                    if sim > SIMILARITY_THRESHOLD:
                        label = cached_label
                        break

                # If not matched, try external match
                if label == "Unknown":
                    match_name = find_matching_user(current_embedding)
                    if match_name:
                        label = match_name
                        embedding_cache[label] = current_embedding

                # Update face tracking info
                last_seen_faces[label] = {
                    'coords': (x, y, w, h),
                    'timestamp': current_time
                }

                # Play audio on first recognition
                # Upload after delay
                upload_delay = UNKNOWN_DELAY if label == "Unknown" else REGISTERED_DELAY
                if current_time - last_upload_time[label] > upload_delay:
                    last_upload_time[label] = current_time
                    if label != "Unknown":
                        deactivate_trip_for_registered_user(label)
                    blob_folder = "unknown" if label == "Unknown" else "registered"
                    blob_path = f"Exit/{blob_folder}/{label}_{uuid.uuid4()}.jpg"
                    _, buffer = cv2.imencode('.jpg', face_crop)
                    if label != "Unknown":
                        spoken_users.add(label)
                        try:
                          play_sound(BEEP_AUDIO_PATH)
                        except Exception as e:
                          logger.error(f"Audio play failed: {e}")
                    try:
                        container_client.upload_blob(name=blob_path, data=buffer.tobytes())
                        log_message = f"{label} face uploaded at {datetime.now()}, Blob: {blob_path}"
                        logger.info(log_message)
                        yield f"data: {log_message}\n\n"
                    except Exception as e:
                        logger.error(f"Error uploading {label} face: {e}")

        except Exception as e:
            logger.error(f"Detection error: {e}")

        # Draw boxes for recently seen faces
        for label, data in list(last_seen_faces.items()):
                x, y, w, h = data['coords']
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                # Draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw label on top
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                current_location="Manglore Bus Stop"
                # Draw current location below
                if current_location:
                    cv2.putText(frame, current_location, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Stream output
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

from flask import Response

# Example: objects assumed to be initialized elsewhere in your app
camera = cv2.VideoCapture(0)  # or your camera ID
# container_client, logger, get_face_embedding, find_matching_user, and current_location should be initialized
current_location="Manglore Bus Stop"

@app.route('/video_feed')
def video_feed():
    return Response(
        detect_faces_and_stream(
            camera=camera,
            container_client=container_client,
            logger=logger,
            get_face_embedding=get_face_embedding,
            find_matching_user=find_matching_user,
            current_location=current_location
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
@app.route('/logs')
def logs():
    def generate_logs():
        # Check if logger has handlers and read from the stream if available
        if logger.handlers:
            handler = logger.handlers[0]  # Get the first handler
            if hasattr(handler, 'stream'):
                # Make sure the stream is available
                for log_message in handler.stream:
                    yield f"data: {log_message}\n\n"
            else:
                yield "data: No stream found in the handler.\n\n"
        else:
            yield "data: Scanning and uploading\n\n"

    return Response(generate_logs(), mimetype='text/event-stream')
@app.route('/delete_images', methods=['POST'])
def delete_images():
    blobs = container_client.list_blobs()
    deleted_count = 0
    for blob in blobs:
        container_client.delete_blob(blob.name)
        deleted_count += 1
    return f"{deleted_count} frame(s) deleted from blob storage."

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False, port=5001)
    camera.release()  # Release the camera when the app stops
