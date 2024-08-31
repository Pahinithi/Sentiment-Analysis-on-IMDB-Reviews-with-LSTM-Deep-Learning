from flask import Flask, request, render_template
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("sentiment_analysis_model.h5")

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()  # Read as a string
    tokenizer = tokenizer_from_json(tokenizer_json)  # Pass the string to tokenizer_from_json

# Initialize history list
history = []

# Define a route for the homepage
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get the movie name and review from the form
        movie_name = request.form.get("movie_name")
        review = request.form.get("review")
        
        # Tokenize and pad the review
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        
        # Make a prediction using the model
        prediction = model.predict(padded_sequence)
        
        # Determine sentiment
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
        
        # Save to history
        history.append({"movie_name": movie_name, "review": review, "sentiment": sentiment})
        
        prediction = f"The sentiment of the review for '{movie_name}' is: {sentiment}"
    
    return render_template("index.html", prediction=prediction)

# Define a route for the history page
@app.route("/history")
def show_history():
    return render_template("history.html", history=history)

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)
