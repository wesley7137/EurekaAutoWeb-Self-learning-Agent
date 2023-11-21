# MultiModalModel Project

## Description
This project aims to create a multi-modal model that integrates text and image-based machine learning models. The project features a set of classes and methods that handle data scraping, relevance scoring, evolutionary search for optimization, and other functionalities.

## Installation
Clone this repository to your local machine. Install the required packages by running:

\`\`\`
pip install -r requirements.txt
\`\`\`

## Usage
To use the MultiModalModel:

1. Import the `MultiModalModel` class.
2. Initialize the `MultiModalModel` object with the required model names and tokenizers for text and image-based models.
3. Use the `run_analysis` method to perform a multi-modal analysis on a given URL, text, and image path.

Example:

\`\`\`python
from MultiModalModel import MultiModalModel

# Initialize
text_model_name = "gpt2"
text_tokenizer_name = "gpt2"
image_model_name = "image_model_name_here"
image_tokenizer_name = "image_tokenizer_name_here"

multi_modal_model = MultiModalModel(text_model_name, text_tokenizer_name, image_model_name, image_tokenizer_name)

# Run analysis
url = "https://example.com"
text = "Sample text here"
image_path = "path/to/image.jpg"

relevance_score, tags, image_category = multi_modal_model.run_analysis(url, text, image_path)
\`\`\`

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
