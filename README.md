# Sentiment Analysis on IMDB Reviews with LSTM Deep Learning

This project implements sentiment analysis on IMDB movie reviews using a Long Short-Term Memory (LSTM) neural network. The model is trained to classify reviews as positive or negative, and the web application is built with Flask. Users can input a movie review, get predictions, and view a history of predictions made during the session.

## Table of Contents
- [Project Overview](#project-overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing Steps](#preprocessing-steps)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [License](#license)

## Project Overview
The goal of this project is to build a sentiment analysis model that predicts whether an IMDB movie review is positive or negative. The project leverages an LSTM neural network for this task, which is particularly effective for processing sequential data like text.

## Demo
You can view a live demo of the application [here]([link-to-demo](https://drive.google.com/file/d/1wjNssZ_Qy-vY1kEfwkbyPXUxqwd6jcD2/view?usp=sharing)).

## Dataset
The dataset used in this project is the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 reviews labeled as positive or negative. The dataset is split into 25,000 training and 25,000 test reviews, ensuring that there is no overlap between the sets.

## Model Architecture
The LSTM model is built using TensorFlow/Keras and consists of the following layers:
- **Embedding Layer**: Maps each word in the review to a dense vector of fixed size.
- **LSTM Layer**: Captures long-term dependencies in the text data.
- **Dense Layer**: Fully connected layer for classification.
- **Output Layer**: Sigmoid activation function to output a probability score between 0 and 1.

### Model Summary:
```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

## Preprocessing Steps
Before training the model, the following preprocessing steps are applied:
1. **Text Cleaning**: Remove HTML tags, non-alphanumeric characters, and convert text to lowercase.
2. **Tokenization**: Convert text into sequences of integers.
3. **Padding**: Pad sequences to ensure uniform input size for the model.

Example of preprocessing code:
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_reviews)

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

# Pad sequences
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length)
test_padded = pad_sequences(test_sequences, maxlen=max_length)
```

## Features
- **Movie Review Sentiment Prediction**: Predict whether a review is positive or negative.
- **Prediction History**: View a history of all predictions made during the session.
- **Beautiful and Responsive UI**: Designed using Bootstrap for a clean and modern look.

## Installation

### Prerequisites
- Python 3.7 or higher
- Flask
- TensorFlow

### Clone the Repository
```bash
git clone https://github.com/Pahinithi/Sentiment-Analysis-on-IMDB-Reviews-with-LSTM-Deep-Learning
cd Sentiment-Analysis-IMDB-Reviews
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python main.py
```
Visit `http://127.0.0.1:5000/` in your web browser to use the application.

## Usage
1. Enter a movie name and review in the provided text fields.
2. Click on the "Predict Sentiment" button.
3. View the sentiment prediction on the same page.
4. Access the prediction history by clicking on the "History" link in the navigation bar.

## Project Structure
```
Sentiment-Analysis-IMDB-Reviews/
│
├── templates/
│   ├── index.html               # Home page of the web application
│   ├── history.html             # Prediction history page
│
├── sentiment_analysis_model.h5  # Pre-trained LSTM model
├── tokenizer.json               # Tokenizer for text preprocessing
├── main.py                      # Main Flask application
├── preprocessing.ipynb          # Jupyter notebook for data preprocessing and model training
├── requirements.txt             # Python dependencies
└── README.md                    # Project README file
```

## Technologies Used
- **Flask**: Micro web framework used for developing the web application.
- **TensorFlow**: Deep learning framework used to build and train the LSTM model.
- **Bootstrap**: Front-end framework for creating responsive and beautiful UI.
- **Jupyter Notebook**: For developing and experimenting with the model.

## Screenshots
<img width="1728" alt="DL12" src="https://github.com/user-attachments/assets/43ad1425-2a87-469f-af68-ff427f60af20">

<img width="1728" alt="DL12-1" src="https://github.com/user-attachments/assets/2311c63d-65dc-4363-a7d5-b4f99cd0431f">



## License
This project is licensed under the MIT License
