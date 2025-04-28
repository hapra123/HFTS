import ast
import json
import numpy as np
from scipy.spatial.distance import cosine
import pyodbc
def get_db_connection():
    server = ''
    database = ''
    username = ''
    password = ''
    driver = '{ODBC Driver 18 for SQL Server}'

    conn_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
    return pyodbc.connect(conn_str)

def find_matching_user():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, face_vector FROM registered_users")
    rows = cursor.fetchall()
    conn.close()

    min_distance = float('inf')
    matched_user = None
    #print(rows)
    for name, stored_vector_str in rows:
        try:
            stored_obj = ast.literal_eval(stored_vector_str)
            print(f"{name} parsed successfully:", stored_obj)
        except Exception as e:
            print(f"Error parsing {name}: {e}")
            print("Offending string:", stored_vector_str)

    


find_matching_user()