# database
import sqlite3


# calling the database module
from Database.database import update_model_database

def fetch_model_names(types=False, subtype=False):
    conn = sqlite3.connect('models.db')
    cursor = conn.cursor()

    if types == 'All Models':
        cursor.execute('''SELECT name FROM models''')
    elif subtype == 'All Models':
        cursor.execute('''SELECT name FROM models WHERE Type = "{}"'''.format(types))
    else:
        cursor.execute('''SELECT name FROM models WHERE Type = "{}" and SubType like "%{}%"'''.format(types, subtype))
        
    model_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return model_names


# Function to retrieve model information from SQLite database
def fetch_model_info(model_name):
    conn = sqlite3.connect('models.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM models WHERE name = ?''', (model_name,))
    model_info = cursor.fetchone()
    conn.close()
    return model_info