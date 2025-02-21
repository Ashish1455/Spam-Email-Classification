# Spam Email Classification

This project is a machine learning-based spam email classifier. It uses natural language processing (NLP) techniques to differentiate between spam and non-spam emails.

## Features
- Preprocessing of email text (removal of stop words, stemming, etc.)
- Training of classification models (Naïve Bayes, Logistic Regression, etc.)
- Evaluation of model performance with accuracy, precision, recall, and F1-score
- Ability to classify new emails as spam or ham

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/Ashish1455/Spam-Email-Classification.git
   cd Spam-Email-Classification
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the classifier:
   ```sh
   python main.py
   ```

## Dataset
- The dataset used contains labeled spam and ham emails for training and testing.

## Usage
- Train a new model or use a pre-trained model to classify emails.
- Modify `config.py` to change model parameters.

## License
This project is open-source and free to use.
