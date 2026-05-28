# 🎬 Movie Genre Classification

A Machine Learning and Natural Language Processing (NLP) based web application that predicts movie genres from plot descriptions using text classification techniques.

---

## Project Overview

This project uses NLP preprocessing and Machine Learning algorithms to classify movie genres based on user-provided movie descriptions or plot summaries.

The application is built using:

* Python
* Flask
* Scikit-learn
* TF-IDF Vectorization
* Machine Learning Classification Models

The system analyzes textual input and predicts the most relevant movie genre.

---

## Features

* Genre prediction from movie plot descriptions
* NLP-based text preprocessing
* TF-IDF feature extraction
* Trained Machine Learning classification model
* Flask web application interface
* Clean and responsive frontend design

---

## Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* HTML
* CSS

---

## Project Structure

```plaintext
movie-genre-classification/
│
├── app.py
├── requirements.txt
├── README.md
├── model.pkl
├── vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── notebooks/
│   └── model_training.ipynb
│
└── dataset/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/manashbaral/movie-genre-classification.git
```

### 2. Navigate to Project Directory

```bash
cd movie-genre-classification
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask server using:

```bash
python app.py
```

Then open your browser and visit:

```plaintext
http://127.0.0.1:5000
```

---

## Machine Learning Workflow

1. Data Collection
2. Text Cleaning and Preprocessing
3. TF-IDF Vectorization
4. Model Training
5. Model Evaluation
6. Flask Integration
7. Genre Prediction

---

## Example Input

```text
A detective investigates mysterious murders occurring in a haunted town while uncovering dark secrets from the past.
```

### Predicted Genre

```text
Thriller / Mystery
```

---

## Future Improvements

* Multi-label genre prediction
* Deep Learning implementation
* Movie poster integration
* Confidence score visualization
* Genre probability graph
* User authentication system
* Deployment on cloud platforms

---

## Contributing

Contributions are welcome.

If you would like to improve this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by Manash Baral
