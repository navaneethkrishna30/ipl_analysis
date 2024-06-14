from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set your secret key for flashing messages

# Function to check if the provided credentials are valid
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_credentials WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template("signup.html")
    elif request.method == 'POST':
        username = request.form.get("user_name")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Check if username or email already exists
        cursor.execute("SELECT * FROM user_credentials WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username or email already exists', category='error')
        else:
            # Insert new user into the database
            cursor.execute("INSERT INTO user_credentials (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            flash('Account created successfully!', category='success')

        conn.close()
        return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")
    elif request.method == 'POST':
        username = request.form.get("user_name")
        password = request.form.get("password")

        user = authenticate_user(username, password)

        if user:
            flash('Login successful!', category='success')
            # Here you can perform additional tasks like setting up session variables
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', category='error')
            return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)