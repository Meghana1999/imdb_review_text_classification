# IMDB Sentiment Text Classification Project

## Introduction
This project is a sentiment analysis application using the IMDB Dataset. It involves training a machine learning model on movie reviews to predict sentiment (positive or negative). The application is built with a Flask backend to serve predictions.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.8  
- Pip (Python package manager)
- Google Collab Notebook (for running `.ipynb` files) , T4 GPU enabled
- Also, you can go through my requirements.txt file inside Flask folder

## Installation

1. **Clone the Repository:**
   - Clone this repository to your local machine.

2. **Install Required Python Libraries:**
   - Open your terminal or command prompt.
   - Navigate to the project directory.
   - Install necessary libraries.  

## Running the Application

1. **Data Preparation and Model Training:**
   - Open the `IMDB_classification_improved.ipynb` file in Google Collab Notebook.
   - Read the csv file by changing the path to your csv file's path.
   - Run all cells in the notebook to train the sentiment analysis model. This process will use `IMDB Dataset.csv` for training. 
   - At the end of the notebook, the model will be saved to a specified path. **Important:** Ensure to modify the path to your desired location for saving the model.

2. **Setting Up the Flask Application:**
   - Navigate to the Flask application folder containing `app.py`.
   - Open `app.py` and ensure the model loading path is updated to where your model is saved.
   - Install Flask using `pip install flask` if not already installed.

3. **Running the Flask Server:**
   - In your terminal or command prompt, navigate to the folder containing `app.py`.
   - Run the command `python app.py` in your terminal to start the Flask server.
   - Once the server is running, open a web browser and go to the URL provided in the terminal.

4. **Using the Application:**
   - The web interface will provide a text box to enter a movie review.
   - Submit the review, and the sentiment (positive or negative) will be displayed based on the trained model's prediction.

## Note
- Ensure all file paths (for the dataset, model saving, and model loading) are correctly set up according to your system's directory structure.
- It's crucial to run the collab Notebook before running the Flask application, as the model needs to be trained and saved first.

## Demo

https://github.com/Meghana1999/imdb_review_text_classification/assets/40660074/864cb4cc-2f8d-4431-82da-fe096307f3ba



  
