#!/usr/bin/env python
import sqlite3

def display_db_contents(db_path, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Query all data from the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Display the rows
        for row in rows:
            print(row)

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the connection to the database
        conn.close()

if __name__ == '__main__':
    # Path to your SQLite database file
    db_path = 'tasks.db'
    # Name of the table you want to display
    table_name = 'tasks'
    display_db_contents(db_path, table_name)
