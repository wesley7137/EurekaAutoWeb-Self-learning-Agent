# Web Scraping Agent Project Outline

## Data Relevance Class
- ~~Initial Relevance Scoring Function~~: Implemented
- ~~User Feedback Integration Function~~: Implemented
- ~~Automated Re-scraping Strategy Function~~: Implemented
- schedule_fine_tuning(): Placeholder added

## Information Categorization Class
- ~~Initial Tagging Function~~: Implemented
- ~~Learning Adaptation Metrics Function~~: Implemented
- ~~User-Driven Categorization~~: Implemented

## Vision Algorithms Class
- ~~Image Analysis~~: Implemented
- ~~Code Extraction~~: Implemented
- Self-Improvement: Placeholder added

## MultiModalModel Class
- Load Model: Placeholder added
- Run Analysis: Placeholder added
- fine_tune_from_checkpoint(): Placeholder added
- memory_mechanism(): Placeholder added

## MemoryMechanism Class
- load_memGPT(): Placeholder added
- memory_update(): Placeholder added

## Additional Thoughts
- Concurrency: Not implemented yet
- Continuous Learning: Not implemented yet
- User Interface: Not implemented yet
- Logging and Monitoring: Logging imports added, but implementation not done

## Asynchronous Model Training System
- Design an asynchronous or parallel system for model training: Not implemented yet
- Fine-tune the model on a scheduled timer (e.g., every 12 hours): Not implemented yet
- Prepare specified directories and files for daily update operations (e.g., `new_text_data.json`, `new_image_data.file`): Not implemented yet
- Use training templates that trigger the fine-tuning task at specified times: Not implemented yet
- Implement a mechanism to load the model from the last checkpoint and fine-tune it further on new data: Not implemented yet
- Explore the use of a higher model (Lora) to hold the fine-tuning logic, while the main model acts as a lower model: Not implemented yet

## Memory Mechanism
- Research and explore the implementation of a memory mechanism for the model: Not implemented yet
- Consider using `memGPT` as a starting point based on recent research: Not implemented yet


Llava1.5 Finetuning notes:
conversations
list
id
string
image
string
[ { "from": "human", "value": "Provide a brief description of the given image.\n<image>" }, { "from": "gpt", "value": "olive oil is a healthy ingredient used liberally ." } ]	
GCC_train_002582585	
GCC_train_002582585.jpg
[ { "from": "human", "value": "<image>\nWrite a terse but informative summary of the picture." }, { "from": "gpt", "value": "3d vector deluxe alphabet of randomly rotated thin golden symbols ." } ]	
GCC_train_002429825	
GCC_train_002429825.jpg
[ { "from": "human", "value": "Share a concise interpretation of the image provided.\n<image>" }, { "from": "gpt", "value": "weathering -- the physical & chemical breakdown of rocks ." } ]	
GCC_train_000190697	
GCC_train_000190697.jpg
