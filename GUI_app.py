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
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class EmailSpamPredictor:
    """Email spam predictor using ensemble voting classifier only."""

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
        """Predict using ensemble voting classifier."""
        try:
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


class EmailSpamGUI:
    """Tkinter GUI for Email Spam Classification"""

    def __init__(self):
        self.predictor = None
        self.window = tk.Tk()
        self.window.title("Email Spam Classifier - Ensemble Model")
        self.window.geometry('700x650')
        self.window.resizable(True, True)

        # Configure style
        self.window.configure(bg='#f0f0f0')

        # Initialize predictor
        try:
            self.predictor = EmailSpamPredictor()
            self.create_widgets()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            self.window.destroy()
            return

        self.window.mainloop()

    def create_widgets(self):
        """Create and arrange GUI widgets"""

        # Title
        title_label = tk.Label(
            self.window,
            text="Email Spam Classifier",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)

        # Model info
        info_text = f"Model: Ensemble Voting Classifier\nLoaded: {self.predictor.model_metadata.get('training_date', 'Unknown')}"
        info_label = tk.Label(
            self.window,
            text=info_text,
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666666'
        )
        info_label.pack(pady=5)

        # Main frame
        main_frame = tk.Frame(self.window, bg='#f0f0f0')
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)

        # Subject input
        tk.Label(main_frame, text="Email Subject:", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(anchor='w')
        self.subject_entry = tk.Entry(main_frame, width=70, font=("Arial", 10))
        self.subject_entry.pack(pady=5, fill='x')

        # Sender input
        tk.Label(main_frame, text="Email Sender:", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(anchor='w',
                                                                                                  pady=(10, 0))
        self.sender_entry = tk.Entry(main_frame, width=70, font=("Arial", 10))
        self.sender_entry.pack(pady=5, fill='x')

        # Email Body input
        tk.Label(main_frame, text="Email Body:", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(anchor='w',
                                                                                                pady=(10, 0))
        self.body_text = scrolledtext.ScrolledText(
            main_frame,
            width=70,
            height=15,
            font=("Arial", 10),
            wrap=tk.WORD
        )
        self.body_text.pack(pady=5, fill='both', expand=True)

        # URLs checkbox
        urls_frame = tk.Frame(main_frame, bg='#f0f0f0')
        urls_frame.pack(pady=10, anchor='w')

        tk.Label(urls_frame, text="Contains URLs:", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(side='left')
        self.urls_var = tk.IntVar()
        self.urls_checkbox = tk.Checkbutton(
            urls_frame,
            text="Yes",
            variable=self.urls_var,
            bg='#f0f0f0',
            font=("Arial", 10)
        )
        self.urls_checkbox.pack(side='left', padx=10)

        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=15)

        # Predict button
        self.predict_button = tk.Button(
            button_frame,
            text="🔍 Predict Spam",
            command=self.predict,
            font=("Arial", 12, "bold"),
            bg='#007acc',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.predict_button.pack(side='left', padx=5)

        # Clear button
        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear All",
            command=self.clear_all,
            font=("Arial", 12, "bold"),
            bg='#cc7a00',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.clear_button.pack(side='left', padx=5)

        # Result frame
        result_frame = tk.Frame(main_frame, bg='#f0f0f0', relief='sunken', bd=2)
        result_frame.pack(pady=10, fill='x')

        tk.Label(result_frame, text="Prediction Result:", font=("Arial", 12, "bold"), bg='#f0f0f0').pack(pady=5)

        self.result_label = tk.Label(
            result_frame,
            text="Enter email details and click 'Predict Spam'",
            font=("Arial", 11),
            bg='#f0f0f0',
            fg='#666666',
            justify='left'
        )
        self.result_label.pack(pady=10)

    def predict(self):
        """Handle prediction button click"""
        subject = self.subject_entry.get().strip()
        sender = self.sender_entry.get().strip()
        body = self.body_text.get('1.0', tk.END).strip()
        urls_flag = self.urls_var.get()

        if not body:
            messagebox.showwarning("Input Error", "Email body is required for prediction.")
            return

        # Disable button during prediction
        self.predict_button.config(state='disabled', text='Predicting...')
        self.window.update()

        try:
            result = self.predictor.predict_email(
                email_text=body,
                sender=sender,
                subject=subject,
                urls=urls_flag
            )

            # Format result display
            if result['prediction'] == 'Spam':
                result_text = f"🚨 SPAM DETECTED!\n"
                result_color = '#cc0000'
            elif result['prediction'] == 'Not Spam':
                result_text = f"✅ NOT SPAM\n"
                result_color = '#00cc00'
            else:
                result_text = f"⚠️ {result['prediction']}\n"
                result_color = '#cc7a00'

            result_text += f"Confidence: {result['confidence']:.1%}\n"
            result_text += f"Model: {result['model_used']}"

            self.result_label.config(
                text=result_text,
                fg=result_color,
                font=("Arial", 12, "bold")
            )

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")
            self.result_label.config(
                text="Prediction failed. Check console for details.",
                fg='#cc0000'
            )

        finally:
            # Re-enable button
            self.predict_button.config(state='normal', text='🔍 Predict Spam')

    def clear_all(self):
        """Clear all input fields"""
        self.subject_entry.delete(0, tk.END)
        self.sender_entry.delete(0, tk.END)
        self.body_text.delete('1.0', tk.END)
        self.urls_var.set(0)
        self.result_label.config(
            text="Enter email details and click 'Predict Spam'",
            fg='#666666',
            font=("Arial", 11)
        )


def main():
    """Launch the GUI application"""
    try:
        EmailSpamGUI()
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()
