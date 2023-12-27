from flask import Flask, request
import pickle

app = Flask(__name__) # инициализируем Flask приложение
pickled_model = pickle.load(open('./model.pkl', 'rb'))
pickled_vectorizer = pickle.load(open('./vectorizerTF.pkl', 'rb'))
@app.route('/')
def hello_world() -> str:
    args = request.args # получаем query аргументы

    data = args.get("data") # получаем query параметр data
    text = [data]
    vec_text = pickled_vectorizer.transform(text)
    prediction = pickled_model.predict(vec_text)
    http_response = str(prediction[0]) # приводит к формату ответа строки

    return http_response
