    
    
import os
import json
import yaml


def setup_directories_and_templates(self):
    # Create directories
    dirs_to_create = [
        self.root_dir,
        self.data_dir,
        self.text_data_dir,
        self.image_data_dir,
        self.model_checkpoint_dir,
        self.text_model_checkpoint_dir,
        self.image_model_checkpoint_dir,
        self.templates_dir
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Create fine-tuning template
    fine_tuning_template = {
        "text_model": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 3
        },
        "image_model": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 3
        }
    }

    with open(self.fine_tuning_template_file, 'w') as f:
        yaml.dump(fine_tuning_template, f)

# Testing the setup
cl = ContinuousLearning()
cl.setup_directories_and_templates()
# Check if the directories and template file have been created properly
os.path.exists(cl.fine_tuning_template_file)