
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List

from typing import List, Dict
import heapq


class InformationCategorization:
    def __init__(self, model_name: str, tokenizer_name: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tags = defaultdict(int)
        self.user_defined_tags = {}
        self.predicted_tags = {}  # To keep track of predicted tags for each article
        self.user_defined_tags = {}  # To store user-defined tags for each article



    def initial_tagging(self, url: str, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_tags = self.model.config.id2label[torch.argmax(logits).item()]
        self.tags[predicted_tags] += 1
        return predicted_tags

    def learning_adaptation_metrics(self) -> Dict[str, float]:
        """
        Use metrics like precision, recall, and F1-score to adjust the tagging algorithm.
        This function returns a dictionary containing these metrics for evaluation.
        """
        all_predicted_tags = []
        all_user_tags = []
        
        for url, p_tags in self.predicted_tags.items():
            u_tags = self.user_defined_tags.get(url, [])
            
            # Here, we assume that tags are transformed into a binary vector representation
            # For demonstration, let's assume there are 10 possible tags, represented by indices 0-9
            predicted_vector = [1 if str(i) in p_tags else 0 for i in range(10)]
            user_vector = [1 if str(i) in u_tags else 0 for i in range(10)]
            
            all_predicted_tags.append(predicted_vector)
            all_user_tags.append(user_vector)

        # Calculate metrics
        precision = precision_score(all_user_tags, all_predicted_tags, average='weighted')
        recall = recall_score(all_user_tags, all_predicted_tags, average='weighted')
        f1 = f1_score(all_user_tags, all_predicted_tags, average='weighted')
        
        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
        
        # These metrics can then be used to adjust the tagging algorithm
        # For demonstration, we're just returning these metrics
        return metrics
    def user_driven_categorization(self, url: str, user_tags: List[str]):
        self.user_defined_tags[url] = user_tags
        for tag in user_tags:
            self.tags[tag] += 1

