
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
encoded_input = tokenizer(text)
output = text_model_name(**encoded_input)
relevance_score 
#image_model_name = "Salesforce/blip-image-captioning-large"
#image_tokenizer_name = "Salesforce/blip-image-captioning-large"

# Initialize model names and tokenizers
url = "https://en.wikipedia.org/wiki/Neuron"
text = "What is a neuron?"

# Initialize your MultiModalModel

#multi_modal_model = (text_model_name, text_tokenizer_name, image_model_name, image_tokenizer_name)

# Run analysis

relevance_score, tags, image_category = multi_modal_model(url, text, image_path)

print(f"Relevance Score: {relevance_score}")
print(f"Tags: {tags}")
print(f"Image Category: {image_category}")
