import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class EmailSpamPredictor:
    """
    Email spam predictor using ensemble voting classifier only.
    """

    def __init__(self, model_path='./models'):
        self.model_path = model_path
        self.ensemble_model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.model_metadata = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Download NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            logger.warning("Could not download NLTK data.")

        self._load_models()

    def _load_models(self):
        """Load ensemble model and preprocessing components."""
        try:
            # Load metadata
            metadata_path = os.path.join(self.model_path, 'model_metadata.pkl')
            if os.path.exists(metadata_path):
                self.model_metadata = joblib.load(metadata_path)
                logger.info("Loaded model metadata")
            else:
                raise FileNotFoundError("Model metadata not found")

            # Load preprocessing components
            self.tfidf_vectorizer = joblib.load(os.path.join(self.model_path, 'tfidf_vectorizer.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_path, 'label_encoder.pkl'))

            # Load ensemble voting classifier
            ensemble_path = os.path.join(self.model_path, 'ensemble_voting.pkl')
            if os.path.exists(ensemble_path):
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info("Loaded ensemble voting classifier")
            else:
                raise FileNotFoundError("Ensemble voting model not found")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_text(self, text):
        """Preprocess text data."""
        if pd.isna(text) or not text:
            return ''

        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())

        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in text.split() if w not in stop_words]
            text = ' '.join(words)
        except:
            logger.warning("Stopwords not available")

        try:
            words = text.split()
            words = [self.stemmer.stem(self.lemmatizer.lemmatize(w)) for w in words]
            text = ' '.join(words)
        except:
            logger.warning("Stemming/Lemmatization not available")

        return text

    def predict_email(self, email_text, sender='', subject='', urls=0):
        """
        Predict using ensemble voting classifier.

        Args:
            email_text (str): Email body
            sender (str): Email sender
            subject (str): Email subject
            urls (int): Binary flag for URLs (0/1)

        Returns:
            dict: Prediction results
        """
        try:
            # Combine text fields (excluding urls from text processing)
            combined_text = f"{subject} {email_text} {sender}"
            processed_text = self.preprocess_text(combined_text)

            if not processed_text.strip():
                logger.warning("No text content after preprocessing")
                return {
                    'prediction': 'Unable to classify',
                    'confidence': 0.0,
                    'label': 'Unknown',
                    'model_used': 'Ensemble_Voting'
                }

            # Transform text to features
            features = self.tfidf_vectorizer.transform([processed_text])

            # Make prediction using ensemble
            prediction = self.ensemble_model.predict(features)[0]

            # Get confidence if available
            if hasattr(self.ensemble_model, 'predict_proba'):
                probabilities = self.ensemble_model.predict_proba(features)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0

            # Convert to label
            label = str(self.label_encoder.inverse_transform([prediction])[0])
            pred_text = 'Spam' if label.lower() in ['spam', '1'] else 'Not Spam'

            result = {
                'prediction': pred_text,
                'confidence': float(confidence),
                'label': label,
                'model_used': 'Ensemble_Voting'
            }

            logger.info(f"Prediction: {pred_text} (Confidence: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'label': 'Error',
                'model_used': 'Ensemble_Voting',
                'error': str(e)
            }

    def get_model_info(self):
        """Get ensemble model information."""
        if self.model_metadata:
            return {
                'best_model_name': 'Ensemble_Voting',
                'training_date': self.model_metadata.get('training_date', 'Unknown'),
                'ensemble_models': self.model_metadata.get('ensemble_models', []),
                'available': True
            }
        return {'available': False, 'message': 'No model loaded'}


def predict_email(text, sender='', subject='', urls=0):
    """
    Convenience function for single email prediction.

    Args:
        text (str): Email body text
        sender (str): Email sender
        subject (str): Email subject
        urls (int): Binary URLs flag

    Returns:
        str: 'Spam' or 'Not Spam'
    """
    try:
        predictor = EmailSpamPredictor()
        result = predictor.predict_email(text, sender, subject, urls)
        return result['prediction']
    except Exception as e:
        logger.error(f"Error in predict_email function: {str(e)}")
        return "Error"


def main():
    """Interactive email spam prediction using ensemble model."""
    print("Email Spam Classifier - Ensemble Voting Model")
    print("=" * 50)

    try:
        predictor = EmailSpamPredictor()
        info = predictor.get_model_info()
        print(f"\nModel: {info['best_model_name']}")
        print(f"Training Date: {info.get('training_date', 'Unknown')}")

        if 'ensemble_models' in info:
            print(f"Ensemble contains: {', '.join(info['ensemble_models'])}")

        print("\nEnter email details:")
        while True:
            print("\n" + "-" * 30)
            subject = input("Email Subject: ").strip()
            sender = input("Email Sender (optional): ").strip()
            body = input("Email Body (in one line): ").strip()
            urls = input("Contains URLs? (0/1): ").strip() or "0"

            if not body:
                print("Email body cannot be empty!")
                continue

            try:
                urls_flag = int(urls)
            except:
                urls_flag = 0

            result = predictor.predict_email(body, sender, subject, urls_flag)

            print("\n" + "=" * 30)
            print("PREDICTION RESULT")
            print("=" * 30)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Model Used: {result['model_used']}")

            if input("\nPredict another email? (y/n): ").strip().lower() != 'y':
                break

        print("\nThank you for using Email Spam Classifier!")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure you have trained the ensemble model first")


if __name__ == "__main__":
    main()
