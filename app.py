from flask import Flask, request, request, render_template
from prediction_function import is_fake 
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET","POST"])
def index():
    status = 2 # 0 - fake, 1 - real, 2 - no prediction yet
    text=""
    if request.method == "POST":
        text = request.form.get("input_text")
        status = is_fake(text)
    return render_template("index.html", text=text, status=status)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
