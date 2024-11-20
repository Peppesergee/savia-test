from flask import Flask, request, jsonify
from SAVIA_retriever.retriever import SAVIARetriever

app = Flask(__name__)

retriever = SAVIARetriever(vector_store_folder = "/SAVIA_vector_stores")

@app.route("/test_API")
def test_API():
    return "Retriever API is running"

@app.route("/retrieve", methods = ['POST'])
def retrieve():

#    question = request.form.get('question')
    request_data = request.get_json()
    question = request_data['question']
#    print(question)

    if request.method == "POST":
        retrieved_item = retriever.create_prompt(question)
        
        return jsonify(retrieved_item)

#    return "<p>Retriever route</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True, port=5001)