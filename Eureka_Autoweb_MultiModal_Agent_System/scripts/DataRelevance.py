
from bs4 import BeautifulSoup
import requests
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple
import random
from collections import Counter
import random
import heapq
from typing import List  # Ensure this line is at the top of your file
import heapq  # Assuming you have imported heapq

class DataRelevance:
    def __init__(self, model_name: str, tokenizer_name: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.relevance_scores = {}
        self.user_feedback = {}  # Store user feedback for each article
        self.best_policy = None  # Best policy found during evolutionary search
        self.best_fitness_score = -float('inf')  # Best fitness score during evolutionary search
        self.REureka = None  # Best reward found during Eureka algorithm
        self.sEureka = float('-inf')  # Best score found during Eureka algorithm
        self.new_temperature_params = {}  # Temperature params after reward reflection
        self.prompt = ""  # Prompt used in Eureka algorithm


    def scrape_website(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        return text

    def initial_relevance_scoring(self, url: str):
        text = self.scrape_website(url)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            relevance_score = torch.sigmoid(logits).item()
        self.relevance_scores[url] = relevance_score
        return relevance_score

    def user_feedback_integration(self, url: str, feedback: float):
        if url in self.relevance_scores:
            self.user_feedback[url] = feedback
            new_score = (self.relevance_scores[url] + feedback) / 2.0
            self.relevance_scores[url] = new_score

    def automated_rescraping_strategy(self) -> List[str]:
        """
        Use historical relevance scores and user feedback to decide which URLs to scrape again.
        This function returns a list of URLs that should be rescraped.
        """
        # Combine relevance scores and user feedback to generate a composite score for each URL
        composite_scores = {}
        for url, relevance_score in self.relevance_scores.items():
            user_feedback = self.user_feedback.get(url, 0)  # Default to 0 if no user feedback is available
            composite_scores[url] = 0.7 * relevance_score + 0.3 * user_feedback  # Weights can be adjusted

        # Identify the top 5 URLs to be rescraped based on their composite scores
        # We use a min heap to keep track of the top 5 URLs
        top_urls = heapq.nlargest(5, composite_scores, key=composite_scores.get)
        
        return top_urls

#region Reward Design Problem
    def reward_design_problem(self, relevance_score: float, user_feedback: float, num_rescrapes: int, adaptation_score: float, user_input: dict, temperature_params: dict, debug=False):
        # Parsing user input to get task specification
        task_spec = user_input  # Assuming user_input is already a parsed dictionary
        task_type = task_spec.get("task_type", "default")
        urgency = task_spec.get("urgency", 0.5)

        relevance_score_weight = 1.0
        user_feedback_weight = 1.0
        num_rescrapes_weight = 1.0
        adaptation_score_weight = 1.0

        if task_type == "urgent" or urgency > 0.7:
            relevance_score_weight *= 1.5  # Adapt weight based on urgency

        # Compute individual reward components
        relevance_score_reward = torch.exp(-temperature_params["relevance_score_temp"] * (1 - relevance_score))
        user_feedback_reward = torch.exp(-temperature_params["user_feedback_temp"] * (1 - user_feedback))
        num_rescrapes_penalty = torch.where(
            torch.tensor(num_rescrapes) > 3,  # Example threshold
            torch.exp(-temperature_params["num_rescrapes_temp"] * (num_rescrapes - 3)),
            torch.zeros(1)
        )
        adaptation_score_reward = torch.exp(-temperature_params["adaptation_score_temp"] * (1 - adaptation_score))

        # Calculate the total reward
        total_reward = (relevance_score_weight * relevance_score_reward +
                        user_feedback_weight * user_feedback_reward +
                        num_rescrapes_weight * num_rescrapes_penalty +
                        adaptation_score_weight * adaptation_score_reward)

        if debug:
            reward_components = {
                "relevance_score_reward": relevance_score_reward.item(),
                "user_feedback_reward": user_feedback_reward.item(),
                "num_rescrapes_penalty": num_rescrapes_penalty.item(),
                "adaptation_score_reward": adaptation_score_reward.item()
            }
            return total_reward.item(), reward_components

        return total_reward.item()
#endregion

#region Fitness Function
    def fitness_function(self, policy: dict, debug=False) -> float:
        # Example metrics to consider for fitness evaluation
        relevance_sum = 0.0
        user_feedback_sum = 0.0
        num_rescrapes_sum = 0.0
        adaptation_score_sum = 0.0

        num_entries = len(policy)

        for action, rewards in policy.items():
            relevance_sum += rewards.get('relevance_score', 0.0)
            user_feedback_sum += rewards.get('user_feedback', 0.0)
            num_rescrapes_sum += rewards.get('num_rescrapes', 0.0)
            adaptation_score_sum += rewards.get('adaptation_score', 0.0)

        # Normalize the metrics
        relevance_avg = relevance_sum / num_entries
        user_feedback_avg = user_feedback_sum / num_entries
        num_rescrapes_avg = num_rescrapes_sum / num_entries
        adaptation_score_avg = adaptation_score_sum / num_entries

        # Compute the fitness score (this can be adapted)
        fitness_score = 0.4 * relevance_avg + 0.3 * user_feedback_avg - 0.2 * num_rescrapes_avg + 0.1 * adaptation_score_avg

        if debug:
            fitness_components = {
                "relevance_avg": relevance_avg,
                "user_feedback_avg": user_feedback_avg,
                "num_rescrapes_avg": num_rescrapes_avg,
                "adaptation_score_avg": adaptation_score_avg
            }
            return fitness_score, fitness_components

        return fitness_score
#endregion


#region Evolutionary Search
    def evolutionary_search(self, num_iterations=10, num_samples=5):
        best_policy = None
        best_fitness_score = -float('inf')
        
        for iteration in range(num_iterations):
            current_policies = []
            current_fitness_scores = []
            
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Generate new samples (policies)
            for _ in range(num_samples):
                # Simulate metrics (these could be obtained from actual data)
                simulated_relevance_score = random.uniform(0.5, 1.0)
                simulated_user_feedback = random.uniform(0.5, 1.0)
                simulated_num_rescrapes = random.uniform(0, 0.7)
                simulated_adaptation_score = random.uniform(0.5, 1.0)
                
                # Create a sample policy based on simulated metrics
                sample_policy = {
                    'action_1': {'relevance_score': simulated_relevance_score, 
                                 'user_feedback': simulated_user_feedback, 
                                 'num_rescrapes': simulated_num_rescrapes, 
                                 'adaptation_score': simulated_adaptation_score}
                    # Add more actions and their metrics
                }
                
                # Compute the fitness score for this policy
                fitness_score = self.fitness_function(sample_policy)
                
                current_policies.append(sample_policy)
                current_fitness_scores.append(fitness_score)
            
            # Find the best policy and fitness score in this iteration
            iteration_best_fitness_score, iteration_best_policy = max(zip(current_fitness_scores, current_policies))
            
            if iteration_best_fitness_score > best_fitness_score:
                best_fitness_score = iteration_best_fitness_score
                best_policy = iteration_best_policy
                
                # Mutate the best policy to use it in the next iteration
                mutated_best_policy = {}
                for action, metrics in best_policy.items():
                    mutated_metrics = {}
                    for key, value in metrics.items():
                        mutation = random.uniform(-0.1, 0.1)
                        new_value = value + mutation
                        new_value = max(0, min(1, new_value))  # Ensure the new value stays within the [0, 1] range
                        mutated_metrics[key] = new_value
                    mutated_best_policy[action] = mutated_metrics
                
                best_policy = mutated_best_policy
        
        return best_fitness_score, best_policy
#endregion

#region Reward Reflection
    """
    The `eureka_algorithm` function uses a reflection-based approach to optimize reward codes for a
    given task and environment.
    
    :param metrics: The `metrics` parameter is a dictionary that contains feedback metrics. It may
    include metrics such as 'success_rate', 'episode_lengths', and 'reward_components'
    :param temperature_params: The `temperature_params` is a dictionary containing the current
    temperature parameters for the reward components. It is used to adjust the temperature of each
    reward component based on the feedback metrics. The keys of the dictionary represent the reward
    components, and the values represent the current temperature for each component
    :param epoch_freq: The `epoch_freq` parameter represents the frequency at which the
    `reward_reflection` function is called during the training process. It determines how often the
    temperature parameters are adjusted based on the feedback metrics
    :return: The function `eureka_algorithm` returns the best reward code (`REureka`) found during the
    algorithm's execution.
    """
    def reward_reflection(self, metrics, temperature_params, epoch_freq):
        """
        Adjust the temperature parameters based on the feedback metrics.
        Args:
        - metrics (dict): A dictionary containing feedback metrics like 'success_rate', 'episode_lengths', etc.
        - temperature_params (dict): A dictionary containing the current temperature parameters for the reward components.
        Returns:
        - new_temperature_params (dict): Updated temperature parameters.
        """
        # Initialize new_temperature_params
        new_temperature_params = {}
        
        # (1) Check if success rates are always near zero
        if metrics.get('success_rate', 1) < 0.01:
            for component, values in metrics.get('reward_components', {}).items():
                if max(values) - min(values) < 0.01:
                    # Adjust temperature_params here (if needed)
                    pass
                
        # (2) Check magnitude differences in reward components
        magnitudes = {component: max(map(abs, values)) for component, values in metrics.get('reward_components', {}).items()}
        max_magnitude = max(magnitudes.values(), default=0)
        
        for component, magnitude in magnitudes.items():
            if magnitude > 10 * (max_magnitude / 11):
                # Adjust temperature_params here (if needed)
                pass
        
        # (3) Update temperature parameters based on success_rate (or other metrics)
        for component, temp in temperature_params.items():
            scaling_factor = 1 + (metrics.get('success_rate', 0.5) - 0.5)
            new_temperature_params[component] = temp * scaling_factor
        
        return new_temperature_params
#endregion


    def eureka_algorithm(self, task_description: str, environment_code: str, initial_prompt: str, N: int, K: int) -> str:
        prompt = initial_prompt
        REureka, sEureka = None, float('-inf')
        for _ in range(N):
            reward_codes = [self.reward_design_problem(None, None) for _ in range(K)]
            scores = [self.fitness_function({'action': reward}) for reward in reward_codes]
            best_index = scores.index(max(scores))
            best_reward = reward_codes[best_index]
            best_score = scores[best_index]
            prompt += f"Reflection({best_reward}, {best_score})"
            if best_score > sEureka:
                REureka, sEureka = best_reward, best_score
        return REureka
