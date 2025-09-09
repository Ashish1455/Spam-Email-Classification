import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib
import warnings
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Could not download NLTK data.")

class EmailSpamClassifier:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = Tokenizer(num_words=5000)

        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None

        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.X_train_seq = None
        self.X_test_seq = None
        self.y_train = None
        self.y_test = None
        self.max_sequence_length = 100

    def load_data(self, filepath):
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        required = ['sender','receiver','date','subject','body','urls','label']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def preprocess_text(self, text):
        if pd.isna(text) or not text:
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.translate(str.maketrans('','',string.punctuation))
        text = ' '.join(text.split())
        try:
            stops = set(stopwords.words('english'))
            words = [w for w in text.split() if w not in stops]
            text = ' '.join(words)
        except:
            logger.warning("Stopwords unavailable")
        try:
            words = [self.stemmer.stem(self.lemmatizer.lemmatize(w)) for w in text.split()]
            text = ' '.join(words)
        except:
            logger.warning("Stemming/Lemmatization unavailable")
        return text

    def feature_engineering(self, df):
        logger.info("Feature engineering...")
        df['combined_text'] = (df['subject'].fillna('') + ' ' +
                               df['body'].fillna('') + ' ' +
                               df['sender'].fillna(''))
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        df['text_length'] = df['processed_text'].str.len()
        df['word_count'] = df['processed_text'].str.split().str.len()
        df['has_attachment'] = df['body'].str.contains('attachment|attached', case=False, na=False)
        try:
            df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
        except:
            df['hour'] = 0
            df['day_of_week'] = 0
        df['url_count'] = df['urls']
        return df

    def prepare_data(self, df):
        y = self.label_encoder.fit_transform(df['label'])
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y)
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train['processed_text'])
        self.X_test_tfidf = self.tfidf_vectorizer.transform(X_test['processed_text'])
        self.tokenizer.fit_on_texts(X_train['processed_text'])
        seq_train = self.tokenizer.texts_to_sequences(X_train['processed_text'])
        seq_test = self.tokenizer.texts_to_sequences(X_test['processed_text'])
        self.X_train_seq = pad_sequences(seq_train, maxlen=self.max_sequence_length)
        self.X_test_seq = pad_sequences(seq_test, maxlen=self.max_sequence_length)
        self.y_train = y_train
        self.y_test = y_test

    def train_traditional_models(self):
        logger.info("Training traditional models...")
        defs = {
            'MultinomialNB': MultinomialNB(),
            'GaussianNB': GaussianNB(),
            'BernoulliNB': BernoulliNB(),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
            'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
            'BaggingClassifier': BaggingClassifier(random_state=42),
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'XGBClassifier': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        for name, model in defs.items():
            try:
                logger.info(f"Training {name}")
                if name == 'GaussianNB':
                    X_tr = self.X_train_tfidf.toarray()
                    X_te = self.X_test_tfidf.toarray()
                else:
                    X_tr = self.X_train_tfidf
                    X_te = self.X_test_tfidf
                model.fit(X_tr, self.y_train)
                y_pred = model.predict(X_te)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_te)[0]
                else:
                    proba = None
                self.models[name] = model
                self.model_scores[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted'),
                    'recall': recall_score(self.y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                    'predictions': y_pred,
                    'probabilities': proba
                }
            except Exception as e:
                logger.error(f"{name} train error: {e}")

    def evaluate_models(self):
        df = pd.DataFrame([
            {'Model': n, **{k:self.model_scores[n][k] for k in ['accuracy','precision','recall','f1_score']}}
            for n in self.model_scores
        ]).sort_values('accuracy', ascending=False)
        hp = df[(df['accuracy']>=0.99)&(df['precision']>=0.99)]
        logger.info("\nModel performance:\n%s", df.to_string(index=False))
        return df, hp

    def plot_results(self):
        """
        Create visualization plots for model performance.
        """
        logger.info("Creating performance visualizations...")

        # Create performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        models = list(self.model_scores.keys())
        accuracies = [self.model_scores[model]['accuracy'] for model in models]
        precisions = [self.model_scores[model]['precision'] for model in models]
        recalls = [self.model_scores[model]['recall'] for model in models]
        f1_scores = [self.model_scores[model]['f1_score'] for model in models]

        # Accuracy plot
        ax1.bar(models, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.95, color='red', linestyle='--', label='99% threshold')
        ax1.legend()

        # Precision plot
        ax2.bar(models, precisions, color='lightgreen')
        ax2.set_title('Model Precision Comparison')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.95, color='red', linestyle='--', label='99% threshold')
        ax2.legend()

        # Recall plot
        ax3.bar(models, recalls, color='lightcoral')
        ax3.set_title('Model Recall Comparison')
        ax3.set_ylabel('Recall')
        ax3.tick_params(axis='x', rotation=45)

        # F1 Score plot
        ax4.bar(models, f1_scores, color='lightyellow')
        ax4.set_title('Model F1-Score Comparison')
        ax4.set_ylabel('F1-Score')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Confusion matrix for best model
        if self.best_model_name:
            y_pred = self.model_scores[self.best_model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Spam', 'Spam'],
                        yticklabels=['Not Spam', 'Spam'])
            plt.title(f'Confusion Matrix - {self.best_model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()

    def save_model(self):
        """
        Save the ensemble model and preprocessing components.
        """
        if 'Ensemble_Voting' not in self.models:
            logger.warning("No ensemble model to save")
            return

        logger.info("Saving ensemble voting classifier")
        os.makedirs('models', exist_ok=True)

        # Save ensemble model only
        joblib.dump(self.models['Ensemble_Voting'], 'models/ensemble_voting.pkl')

        # Save preprocessing components
        joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.tokenizer, 'models/tokenizer.pkl')

        # Save metadata with ensemble info
        metadata = {
            'best_model_name': 'Ensemble_Voting',
            'model_scores': self.model_scores,
            'training_date': datetime.now().isoformat(),
            'max_sequence_length': self.max_sequence_length,
            'ensemble_models': [name for name in self.model_scores.keys() if name != 'Ensemble_Voting']
        }

        joblib.dump(metadata, 'models/model_metadata.pkl')
        logger.info("Ensemble model and components saved successfully")

    def train_and_evaluate(self, data_path):
        df = self.load_data(data_path)
        df = self.feature_engineering(df)
        self.prepare_data(df)
        self.train_traditional_models()
        results_df, high_perf = self.evaluate_models()

        # Build ensemble
        top = high_perf['Model'].tolist()
        estimators = [(n,self.models[n]) for n in top]
        if estimators:
            vc = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            vc.fit(self.X_train_tfidf, self.y_train)
            yp = vc.predict(self.X_test_tfidf)
            self.models['Ensemble_Voting'] = vc
            self.model_scores['Ensemble_Voting'] = {
                'accuracy': accuracy_score(self.y_test,yp),
                'precision': precision_score(self.y_test,yp,average='weighted'),
                'recall': recall_score(self.y_test,yp,average='weighted'),
                'f1_score': f1_score(self.y_test,yp,average='weighted'),
                'predictions': yp
            }
            logger.info("Ensemble_Voting - Acc: %.4f, Prec: %.4f",
                        self.model_scores['Ensemble_Voting']['accuracy'],
                        self.model_scores['Ensemble_Voting']['precision'])
        else:
            logger.warning("No models ≥99%% for ensemble")

        # Continue with plotting and saving
        self.plot_results()
        self.save_model()
        return results_df, high_perf

def main():
    clf = EmailSpamClassifier()
    data_path = 'data/CEAS_08.csv'
    if not os.path.exists(data_path):
        logger.error("Data not found")
        return
    clf.train_and_evaluate(data_path)

if __name__=='__main__':
    main()
