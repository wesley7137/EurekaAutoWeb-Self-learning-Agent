from DataRelevance import DataRelevance
from InformationCategorization import InformationCategorization
from VisionFunctions import VisionFunctions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForImageClassification
import sqlite3

class MultiModalModel:
    def __init__(self, text_model_name: str, text_tokenizer_name: str, image_model_name: str, image_tokenizer_name: str):
        super().__init__(text_model_name, text_tokenizer_name, image_model_name, image_tokenizer_name)
        self.data_relevance = DataRelevance(text_model_name, text_tokenizer_name)
        self.information_categorization = InformationCategorization(text_model_name, text_tokenizer_name)
        self.vision_functions = VisionFunctions(image_model_name, image_tokenizer_name)
        self.best_policy = None  # Best policy found during evolutionary search
        self.best_fitness_score = -float('inf')  # Best fitness score during evolutionary search
        self.REureka = None  # Best reward found during Eureka algorithm
        self.sEureka = float('-inf')  # Best score found during Eureka algorithm
        self.new_temperature_params = {}  # Temperature params after reward reflection
        self.prompt = ""  # Prompt used in Eureka algorithm
        self.data_relevance = DataRelevance(text_model_name, text_tokenizer_name)
        self.information_categorization = InformationCategorization(text_model_name, text_tokenizer_name)
        self.vision_functions = VisionFunctions(image_model_name, image_tokenizer_name)
        self.init_db()  # Initialize the database
        self.save_metric("relevance_score", relevance_score)
        self.save_metric("tags", ','.join(tags))
        self.save_metric("image_category", image_category)
        self.best_fitness_score = self.data_relevance.evolutionary_search()

    def init_db(self):
        # Create a new SQLite database or connect to an existing one
        self.conn = sqlite3.connect('multi_modal_model.db')
        
        # Create tables
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
        
        print("Database initialized and tables created.")

    def load_model(self):
        # Fetch the latest model checkpoint from the database
        latest_checkpoint = self.get_latest_model_checkpoint("text_model")
        if latest_checkpoint:
            self.text_model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint)
        else:
            self.text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        
        latest_checkpoint = self.get_latest_model_checkpoint("image_model")
        if latest_checkpoint:
            self.image_model = AutoModelForImageClassification.from_pretrained(latest_checkpoint)
        else:
            self.image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
        self.image_tokenizer = AutoTokenizer.from_pretrained(image_tokenizer_name)

    def run_analysis(self, url: str, text: str, image_path: str):
        relevance_score = self.data_relevance.initial_relevance_scoring(url)
        tags = self.information_categorization.initial_tagging(url, text)
        image_category = self.vision_functions.image_analysis(url, image_path)
        
        # Save metrics into database
        
        return relevance_score, tags, image_category

    def save_metric(self, metric_name, value):
        with self.conn:
            self.conn.execute("INSERT INTO metrics (metric_name, value) VALUES (?, ?)", (metric_name, value))
        print(f"Saved metric {metric_name} with value {value} into the database.")
        
    def save_model_checkpoint(self, model_name, checkpoint_path):
        with self.conn:
            self.conn.execute("INSERT INTO model_checkpoints (model_name, checkpoint_path) VALUES (?, ?)", (model_name, checkpoint_path))
        print(f"Saved model checkpoint for {model_name} at {checkpoint_path} into the database.")
        
    def get_metrics(self, metric_name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT value, timestamp FROM metrics WHERE metric_name = ?", (metric_name,))
        return cursor.fetchall()
    
    def get_latest_model_checkpoint(self, model_name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT checkpoint_path FROM model_checkpoints WHERE model_name = ? ORDER BY timestamp DESC LIMIT 1", (model_name,))
        return cursor.fetchone()