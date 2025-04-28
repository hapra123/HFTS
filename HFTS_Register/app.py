import os
import pyodbc
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from io import BytesIO
from PIL import Image
from deepface import DeepFace
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Database connection
def get_db_connection():
    server = ''
    database = ''
    username = ''
    password = ''
    driver = '{ODBC Driver 18 for SQL Server}'
    
    conn_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    return pyodbc.connect(conn_str)

# Azure Blob config
connection_str = ""
container_name = "faceframes"

# Connect to Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(connection_str)
container_client = blob_service_client.get_container_client(container_name)

# Dummy face embedding function (simulate)
def get_face_embedding(image_bytes):
    # Open the image
    img = Image.open(BytesIO(image_bytes))
    img_array = np.array(img)
    
    # Get the face representation
    embeddings = DeepFace.represent(img_array, model_name="Facenet", enforce_detection=False)

    # Keep only the 'embedding' key from each item
    if isinstance(embeddings, list):
        return [{'embedding': entry['embedding']} for entry in embeddings]
    elif isinstance(embeddings, dict) and 'embedding' in embeddings:
        return {'embedding': embeddings['embedding']}
    else:
        return []  # In case embeddings is not a list

# Delete all frames (blobs) in the container
def delete_all_frames():
    blobs = container_client.list_blobs()  # Get list of all blobs in the container
    for blob in blobs:
        print(f"Deleting {blob.name}...")
        container_client.delete_blob(blob.name)  # Delete each blob
    print("All frames have been deleted.")

# Delete all users from the database

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_user():
    name = request.form['name']
    image_file = request.files['image']
    balance = request.form['balance']

    # Convert image to bytes
    img = Image.open(image_file)
    image_bytes = BytesIO()
    img.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Upload image to Azure Blob Storage under "registered_faces/"
    blob_name = f"registered_faces/{name.replace(' ', '_')}.png"  # Use name as filename
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(image_bytes.getvalue(), overwrite=True)

    # Reset image bytes for DeepFace
    image_bytes.seek(0)
    face_vec = get_face_embedding(image_bytes.read())

    # Store in DB
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO registered_users (name, face_vector, balance) VALUES (?, ?, ?)", name, str(face_vec), balance)
    conn.commit()
    conn.close()

    message = "User registered and image stored successfully!"
    message_type = "success"
    return render_template('index.html', message=message, message_type=message_type)

@app.route('/delete_all', methods=['POST'])
def delete_all():
    try:
        # Call the functions to delete blobs and users
        delete_all_frames()
        # Success message
        message = "All blobs have been deleted successfully!"
        message_type = "success"
    except Exception as e:
        # Error message
        message = f"Error: {str(e)}"
        message_type = "error"

    return render_template('index.html', message=message, message_type=message_type)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
