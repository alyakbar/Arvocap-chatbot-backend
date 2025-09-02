import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Optional
import logging
from config import OPENAI_API_KEY, CHAT_MODEL, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from vector_database import ChatbotKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotTrainer:
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self.knowledge_base = ChatbotKnowledgeBase()
        
        if use_openai:
            openai.api_key = OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.setup_local_model()

    def update_openai_api_key(self, new_key: str):
        """Update the OpenAI API key for training operations."""
        if not new_key:
            return
        try:
            openai.api_key = new_key
            # Recreate client with new key
            self.client = openai.OpenAI(api_key=new_key)
            logger.info("Updated OpenAI API key for ChatbotTrainer")
        except Exception as e:
            logger.error(f"Failed to update OpenAI API key in ChatbotTrainer: {e}")
    
    def setup_local_model(self):
        """Setup local model for training"""
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_training_data(self, processed_data_file: str) -> List[Dict]:
        """Prepare training data from processed file"""
        try:
            with open(processed_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            training_examples = []
            
            # Convert Q&A pairs to training format
            for qa in data.get('qa_pairs', []):
                training_examples.append({
                    'input': qa['question'],
                    'output': qa['answer'],
                    'keywords': qa.get('keywords', [])
                })
            
            # Create synthetic conversations from documents
            for doc in data.get('documents', [])[:50]:  # Limit for demo
                content = doc['content']
                if len(content) > 100:
                    # Generate potential questions about the content
                    potential_questions = self._generate_questions_from_content(content)
                    for question in potential_questions:
                        training_examples.append({
                            'input': question,
                            'output': content[:500],  # Truncate response
                            'keywords': doc.get('keywords', [])
                        })
            
            logger.info(f"Prepared {len(training_examples)} training examples")
            return training_examples
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return []
    
    def _generate_questions_from_content(self, content: str) -> List[str]:
        """Generate potential questions from content"""
        # Simple question generation based on keywords
        questions = []
        
        # Extract key phrases and create questions
        sentences = content.split('.')[:3]  # First 3 sentences
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                questions.append(f"What is {sentence.strip().lower()}?")
                questions.append(f"Can you explain {sentence.strip().lower()}?")
        
        return questions[:5]  # Limit questions
    
    def create_openai_training_file(self, training_examples: List[Dict], 
                                   output_file: str = "training_data.jsonl"):
        """Create training file for OpenAI fine-tuning"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in training_examples:
                    training_obj = {
                        "messages": [
                            {"role": "user", "content": example['input']},
                            {"role": "assistant", "content": example['output']}
                        ]
                    }
                    f.write(json.dumps(training_obj) + '\n')
            
            logger.info(f"Created OpenAI training file: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error creating training file: {e}")
            return None
    
    def fine_tune_openai_model(self, training_file: str):
        """Fine-tune OpenAI model"""
        try:
            # Upload training file
            with open(training_file, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            logger.info(f"Uploaded training file: {file_id}")
            
            # Create fine-tuning job
            fine_tune_response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model="gpt-3.5-turbo"
            )
            
            job_id = fine_tune_response.id
            logger.info(f"Started fine-tuning job: {job_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error fine-tuning OpenAI model: {e}")
            return None
    
    def train_local_model(self, training_examples: List[Dict]):
        """Train local model using Transformers"""
        try:
            # Prepare dataset
            train_texts = []
            for example in training_examples:
                conversation = f"User: {example['input']}\nBot: {example['output']}"
                train_texts.append(conversation)
            
            # Tokenize
            train_encodings = self.tokenizer(
                train_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create dataset
            class ChatDataset(Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings
                
                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    return item
                
                def __len__(self):
                    return len(self.encodings['input_ids'])
            
            train_dataset = ChatDataset(train_encodings)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                learning_rate=LEARNING_RATE,
                save_steps=1000,
                save_total_limit=2,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting local model training...")
            trainer.train()
            
            # Save model
            self.model.save_pretrained('./trained_model')
            self.tokenizer.save_pretrained('./trained_model')
            
            logger.info("Local model training completed")
            
        except Exception as e:
            logger.error(f"Error training local model: {e}")

class ChatbotInterface:
    def __init__(self, use_openai: bool = True, model_path: Optional[str] = None):
        self.use_openai = use_openai
        self.knowledge_base = ChatbotKnowledgeBase()
        
        if use_openai:
            openai.api_key = OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = CHAT_MODEL
        else:
            self.load_local_model(model_path or './trained_model')

    def set_openai_api_key(self, new_key: str):
        """Update the OpenAI API key for chat operations at runtime."""
        if not new_key:
            return
        try:
            openai.api_key = new_key
            self.client = openai.OpenAI(api_key=new_key)
            logger.info("Updated OpenAI API key for ChatbotInterface")
        except Exception as e:
            logger.error(f"Failed to update OpenAI API key in ChatbotInterface: {e}")
    
    def load_local_model(self, model_path: str):
        """Load locally trained model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            logger.info(f"Loaded local model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            # Fallback to base model
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
    def generate_response(self, user_input: str) -> str:
        """Generate response to user input"""
        try:
            # Get relevant context from knowledge base
            context = self.knowledge_base.find_relevant_context(user_input)
            
            if self.use_openai:
                return self._generate_openai_response(user_input, context)
            else:
                return self._generate_local_response(user_input, context)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def _generate_openai_response(self, user_input: str, context: str) -> str:
        """Generate response using OpenAI API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful chatbot. Use the following context to answer questions: {context}"
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return "I'm having trouble connecting to the AI service."
    
    def _generate_local_response(self, user_input: str, context: str) -> str:
        """Generate response using local model"""
        try:
            # Prepare input with context
            prompt = f"Context: {context}\n\nUser: {user_input}\nBot:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract bot response
            bot_response = response.split("Bot:")[-1].strip()
            
            return bot_response
            
        except Exception as e:
            logger.error(f"Error with local model: {e}")
            return "I'm having trouble processing your request."

def main():
    """Example usage"""
    print("Chatbot Training System")
    print("-" * 30)
    
    choice = input("Choose mode:\n1. Train chatbot\n2. Chat with bot\nEnter choice (1-2): ")
    
    if choice == "1":
        # Training mode
        trainer = ChatbotTrainer(use_openai=True)
        
        # Prepare training data
        training_examples = trainer.prepare_training_data('processed_data.json')
        
        if training_examples:
            # Create training file for OpenAI
            training_file = trainer.create_openai_training_file(training_examples)
            
            if training_file:
                print(f"Training file created: {training_file}")
                print("You can now upload this to OpenAI for fine-tuning")
            
            # Optionally train local model
            local_choice = input("Train local model too? (y/n): ")
            if local_choice.lower() == 'y':
                trainer.train_local_model(training_examples)
    
    elif choice == "2":
        # Chat mode
        use_openai = input("Use OpenAI (y) or local model (n)? ").lower() == 'y'
        chatbot = ChatbotInterface(use_openai=use_openai)
        
        print("\nChatbot ready! Type 'quit' to exit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            
            response = chatbot.generate_response(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    main()
