# Importing Libraries
import sqlite3
from Database.model_details import models_info

def table_exists(cursor):
    """Check if the 'models' table exists in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='models'")
    return cursor.fetchone() is not None

def update_record(cursor, model_name, model_data):
    """Update an existing record in the 'models' table."""
    cursor.execute('''UPDATE models SET description=?, type=?, subtype=?, usage_scenarios=?, advantages=?, limitations=?, data_requirements=?, example=?,render=?,documentation=?, implementation_code=? WHERE name=?''',
                   ('\n'.join(model_data['Description']),
                    '\n'.join(model_data['Type']),
                    '\n'.join(model_data['SubType']),
                    '\n'.join(model_data['Usage Scenarios']),
                    '\n'.join(model_data['Advantages']),
                    '\n'.join(model_data['Limitations']),
                    '\n'.join(model_data['Data Requirements']),
                    '\n'.join(model_data['Example']),
                    '\n'.join(model_data['Render']),
                    '\n'.join(model_data['Documentation']),
                    model_data['Implementation Code'],
                    model_name))

def insert_record(cursor, model_name, model_data):
    """Insert a new record into the 'models' table."""
    cursor.execute('''INSERT INTO models (name, description,type, subtype, usage_scenarios, advantages, limitations,data_requirements, example, render,documentation, implementation_code)
                      VALUES (?, ?, ?,?,?, ?,?, ?,?,?,?, ?)''',
                   (model_name,
                    '\n'.join(model_data['Description']),
                    '\n'.join(model_data['Type']),
                    '\n'.join(model_data['SubType']),
                    '\n'.join(model_data['Usage Scenarios']),
                    '\n'.join(model_data['Advantages']),
                    '\n'.join(model_data['Limitations']),
                    '\n'.join(model_data['Data Requirements']),
                    '\n'.join(model_data['Example']),
                    '\n'.join(model_data['Render']),
                    '\n'.join(model_data['Documentation']),
                    model_data['Implementation Code']))

def update_model_database(database_file):
    # Create SQLite database and connect to it
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Check if the database already exists
    if table_exists(cursor):
        # Database already exists, update it with new model information
        for model_name, model_data in models_info.items():
            if cursor.execute('''SELECT 1 FROM models WHERE name = ?''', (model_name,)).fetchone():
                update_record(cursor, model_name, model_data)
            else:
                insert_record(cursor, model_name, model_data)
    else:
        # Database doesn't exist, create it and populate with model information
        cursor.execute('''CREATE TABLE IF NOT EXISTS models (
                            id INTEGER PRIMARY KEY,
                            name TEXT,
                            type TEXT,
                            subtype TEXT,
                            description TEXT,
                            usage_scenarios TEXT,
                            advantages TEXT,
                            limitations TEXT,
                            data_requirements TEXT,
                            example TEXT,
                            render TEXT,
                            documentation TEXT,
                            implementation_code TEXT
                        )''')
        # Insert model information into the database
        for model_name, model_data in models_info.items():
            insert_record(cursor, model_name, model_data)

    # Commit changes and close connection
    conn.commit()
    conn.close()
