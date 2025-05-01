from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.db_initialized = False  # прапорець для ініціалізації БД

# --------------- ІНІЦІАЛІЗАЦІЯ БАЗИ ДАНИХ ----------------
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.before_request
def initialize_db_once():
    if not app.db_initialized:
        init_db()
        app.db_initialized = True

# ------------------ РЕЄСТРАЦІЯ -----------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Користувач з таким ім'ям вже існує!"
    return render_template('register.html')

# ------------------ АВТОРИЗАЦІЯ -----------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Невірне ім’я користувача або пароль!"
    return render_template('login.html')

# ------------------ ВИХІД -----------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ------------------ ГОЛОВНА СТОРІНКА -----------------------
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    image_path = "static/image.png"  # ваш метод генерації графіка
    return render_template('index.html', image_path=image_path)

# ------------------ ЗАПУСК -----------------------
if __name__ == '__main__':
    app.run(debug=True)
