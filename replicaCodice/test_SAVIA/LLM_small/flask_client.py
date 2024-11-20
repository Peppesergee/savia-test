from flask import Flask, request, jsonify
import os
from Llama_3_2_3B_Instruct.utils_LLM.helper_functions import load_configs
from Llama_3_2_3B_Instruct.predictor.predictor import Predictor

app = Flask(__name__)

configs_LLM = load_configs(configs_path = os.path.join("Llama_3_2_3B_Instruct", "./configs/configs.yml"))

predictor = Predictor(configs_LLM)

@app.route("/test_API")
def test_API():
    return "LLM API is running"

@app.route("/generate", methods = ['POST'])
def generate():
    request_data = request.get_json()
    
    if request.method == "POST":
        pred = predictor.inference(request_data, use_streamer = False)

    return jsonify({"pred": pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False, port=5000)