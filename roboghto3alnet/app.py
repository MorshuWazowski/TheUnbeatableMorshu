from flask import Flask, render_template, jsonify
from threading import Lock
import random

app = Flask(__name__)
app_lock = Lock()
current_move = "waiting"
taunts = ["Too slow!", "I see your moves!", "Try harder!"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_move')
def get_move():
    with app_lock:
        return jsonify(move=current_move)

@app.route('/set_move/<new_move>')
def set_move(new_move):
    global current_move
    with app_lock:
        current_move = new_move
        return jsonify(
            move=new_move,
            taunt=random.choice(taunts)
        )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)