from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods = ["GET", "POST"])
def home():
    prediction = None
    confidence = None
    if request.method == "POST":
        article = request.form["article"]
        article_vector = vectorizer.transform([article])
        prediction = model.predict(article_vector)[0]
        if prediction == 0:
            prediction = "Fake"
        else:
            prediction = "Real"
        probs = model.predict_proba(article_vector)[0] #confidence score
        confidence = round(max(probs) * 100, 2)
    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence)
if __name__ == "__main__":
    app.run(debug = True)  