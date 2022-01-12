import flask
import threading
import time
from flask import Flask, render_template, make_response, redirect, request, current_app, send_from_directory
from flask_restful import Resource, Api
from load_model import ModelPredict
import os


app = Flask(__name__)
api = Api(app)

try:
    @app.route("/")
    def redirHome():
        return redirect("/home", code=302)

    @app.route("/home")
    def renderHome():
        return render_template("home.html")

    @app.route("/prediction-module/predictform")
    def renderForm():
        return render_template("predictform.html")

    app.config["DOWNLOAD_FOLDER"] = "./CryptoAI/CryptoModelData/PermData/"

    @app.route("/downloads/<path:filename>")
    def download(filename):
        downloads = os.path.join(current_app.root_path, app.config['DOWNLOAD_FOLDER'])
        return send_from_directory(directory=downloads, filename=filename)

    @app.route("/prediction-module/prediction", methods=["GET", "POST"])
    def predictResults():
        if request.method == "GET":
            return redirect("/prediction-inability")
        if request.method == "POST":
            form_data = request.form
            try:
                predict = [form_data["timestamps0"], form_data["timestamps1"]]
                file = [form_data["market"], form_data["crypto"]]
                results = ModelPredict(file, predict)
                return render_template("predictions.html", data=results)
            except:
                return render_template("Something-Wrong.html")



    if __name__ == "__main__":
        app.run()
except Exception as e:
    print(e)
