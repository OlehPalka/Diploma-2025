import mysql.connector

# Connect to MySQL server (Make sure MySQL is running)
conn = mysql.connector.connect(
    host="127.0.0.1",  # or "127.0.0.1"
    user="root",       # Change this if using a different user
    password="12345678",
    port=3306 # Set your MySQL root password
)

# Create a cursor object
cursor = conn.cursor()

# Create a new database
db_name = "matomo_part"
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

print(f"Database '{db_name}' created successfully!")

# Close the connection
cursor.close()
conn.close()
