from azure.storage.blob import BlobServiceClient

# Azure Blob config
connection_str = ""
container_name = "faceframes"

# Connect to Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(connection_str)
container_client = blob_service_client.get_container_client(container_name)

def delete_all_frames():
    blobs = container_client.list_blobs()  # Get list of all blobs in the container
    for blob in blobs:
        print(f"Deleting {blob.name}...")
        container_client.delete_blob(blob.name)  # Delete each blob
    print("All frames have been deleted.")

# Call the function to delete all frames
delete_all_frames()
