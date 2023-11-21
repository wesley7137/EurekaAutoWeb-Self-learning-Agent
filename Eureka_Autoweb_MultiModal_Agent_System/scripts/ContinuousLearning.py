# Implementing the ContinuousLearning class with working function code
import schedule
import time
import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForImageClassification, TrainingArguments, Trainer


class ContinuousLearning:
    def __init__(self, text_model_name, text_tokenizer_name, image_model_name):
        self.root_dir = "MultiModalSystem"
        self.data_dir = os.path.join(self.root_dir, "Data")
        self.text_data_dir = os.path.join(self.data_dir, 'D:\\PROJECTS\\Eureka_AutoWeb_Agent\\MultiModalSystem\\Data\\Text\\new_text_data.json')
        self.image_data_dir = os.path.join(self.data_dir, "D:\\PROJECTS\\Eureka_AutoWeb_Agent\\MultiModalSystem\\Data\\Images\\new_image_data.file")
        self.model_checkpoint_dir = os.path.join(self.root_dir, "ModelCheckpoints")
        self.text_model_checkpoint_dir = os.path.join(self.model_checkpoint_dir, "TextModel")
        self.image_model_checkpoint_dir = os.path.join(self.model_checkpoint_dir, "ImageModel")
        self.templates_dir = os.path.join(self.root_dir, "Templates")
        self.fine_tuning_template_file = os.path.join(self.templates_dir, "fine_tuning_template.yaml")
        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
        self.fine_tune_config = "fine_tuning_template.yaml"

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def init_directories(self):
        # Create directories if they don't exist
        if not os.path.exists('daily_updates'):
            os.makedirs('daily_updates')

    def schedule_fine_tuning(self):
        # Schedule the fine-tuning task to run every 12 hours
        schedule.every(12).hours.do(self.fine_tune_model)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

    def fine_tune_model(self):
        print("Fine-tuning the model...")
        
        # Load the latest checkpoint if exists
        self.load_latest_checkpoint()
        
        # Read new data from text_data_path and image_data_path
        with open(self.text_data_path, 'r') as f:
            new_text_data = json.load(f)
        
        # TODO: Load new_image_data from self.image_data_path
        
        # Fine-tuning the text model (Here we need to have a Dataset object, which is not implemented yet)
        # training_args = TrainingArguments(
        #     output_dir='./results', 
        #     overwrite_output_dir=True,
        #     num_train_epochs=1,
        #     per_device_train_batch_size=32,
        #     save_steps=10_000,
        #     save_total_limit=2,
        # )
        # trainer = Trainer(
        #     model=self.text_model,
        #     args=training_args,
        #     train_dataset=new_text_data,  # this needs to be a Dataset object
        # )
        # trainer.train()
        
        # TODO: Fine-tune the image model

        # Save new checkpoint
        self.save_checkpoint()


    def load_latest_checkpoint(self):
        if self.latest_checkpoint:
            self.text_model.load_state_dict(torch.load(self.latest_checkpoint))
            print(f"Loaded the model from the latest checkpoint {self.latest_checkpoint}")


    def save_checkpoint(self):
        checkpoint_path = f'checkpoints/checkpoint_{time.time()}.pth'
        torch.save(self.text_model.state_dict(), checkpoint_path)
        self.latest_checkpoint = checkpoint_path
        print(f"Saved the current model state as a new checkpoint at {checkpoint_path}")


    def fine_tune_text_model(self, new_data_path):
        # Define the TrainingArguments
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            save_steps=10_000,
            save_total_limit=2,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.text_model,
            args=training_args,
            train_dataset=new_data_path,  # This should be a Dataset object
            # add more arguments here if needed
        )
        
        # Fine-tune the model
        trainer.train()
        
        
    def fine_tune_image_model(self, new_data_path):
        # Assume we have a similar Trainer setup for image model
        # Fine-tuning code for the image model would go here
        pass


    def save_model_checkpoint(self):
        # Save checkpoints
        self.text_model.save_pretrained(f"{self.checkpoint_dir}text_model/")
        self.image_model.save_pretrained(f"{self.checkpoint_dir}image_model/")
        
        
    def load_last_checkpoint(self):
        # Load last checkpoints
        self.text_model.from_pretrained(f"{self.checkpoint_dir}text_model/")
        self.image_model.from_pretrained(f"{self.checkpoint_dir}image_model/")


    def read_fine_tune_template(self):
        with open(self.fine_tune_config, 'r') as f:
            config = yaml.safe_load(f)
        return config


    def apply_fine_tuning(self):
        # Read the fine-tuning template
        config = self.read_fine_tune_template()
        
        # Fine-tune text model
        self.fine_tune_text_model(config['text_model']['new_data_path'])
        
        # Fine-tune image model
        self.fine_tune_image_model(config['image_model']['new_data_path'])
        
        # Save the model checkpoint
        self.save_model_checkpoint()


    def schedule_fine_tuning(self):
        # Load the last checkpoint
        self.load_last_checkpoint()
        
        # Schedule the fine-tuning task
        schedule.every(12).hours.do(self.apply_fine_tuning)

        while True:
            schedule.run_pending()
            time.sleep(1)
