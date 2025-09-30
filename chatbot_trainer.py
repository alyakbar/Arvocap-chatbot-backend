import openai
import asyncio
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Optional
import logging
import re
from config import OPENAI_API_KEY, CHAT_MODEL, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from vector_database import ChatbotKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Limit concurrent embedding/context computations to avoid CPU thrash under load
EMBEDDING_MAX_CONCURRENCY = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4"))
EMBEDDING_SEMAPHORE = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)

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
    
    def train_on_data(self, processed_data_file: str) -> bool:
        """
        Train the chatbot on processed data
        For OpenAI-based systems, this primarily updates the vector database
        """
        try:
            logger.info(f"Training chatbot on data from: {processed_data_file}")
            
            # For OpenAI-based training, we mainly need to ensure the 
            # vector database is updated with the latest data
            if self.use_openai:
                # The vector database has already been updated in the pipeline
                # This is essentially a no-op for OpenAI since we use embeddings
                logger.info("✅ OpenAI-based training completed (vector database updated)")
                return True
            else:
                # For local models, perform actual training
                training_examples = self.prepare_training_data(processed_data_file)
                if training_examples:
                    self.train_local_model(training_examples)
                    logger.info("✅ Local model training completed")
                    return True
                else:
                    logger.error("❌ No training examples found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in train_on_data: {e}")
            return False

class ChatbotInterface:
    def __init__(self, use_openai: bool = True, model_path: Optional[str] = None, knowledge_base: Optional[ChatbotKnowledgeBase] = None):
        self.use_openai = use_openai
        # Reuse a shared knowledge base if provided to avoid duplicate model loads
        self.knowledge_base = knowledge_base or ChatbotKnowledgeBase()
        # Compile regex once for sanitization of the legacy email 'invest@arvocap.com'
        # Requirement: change 'invest@arvocap.com' -> 'clients@arvocap.com' in bot responses
        self._legacy_email_pattern = re.compile(r"\binvest@arvocap\.com\b", re.IGNORECASE)
        self._new_email = "clients@arvocap.com"
        
        if use_openai:
            openai.api_key = OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            # Async client for non-blocking requests
            try:
                self.async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            except Exception:
                self.async_client = None
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
            # Refresh async client too if available
            try:
                self.async_client = openai.AsyncOpenAI(api_key=new_key)
            except Exception:
                self.async_client = None
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
        """Generate response to user input with citations when possible"""
        try:
            # Get top relevant docs with metadata for citations
            top_results = self.knowledge_base.search_similar_content(user_input, max_results=3)
            # Build context (concatenate trimmed content)
            ctx_parts = []
            for r in top_results:
                content = (r.get('content') or '').strip()
                if content:
                    ctx_parts.append(content[:800])
            context = "\n\n".join(ctx_parts)

            # Build references list for citation rendering
            references = []
            for i, r in enumerate(top_results, start=1):
                md = r.get('metadata') or {}
                src_type = md.get('type') or ''
                url = (md.get('url') or '').strip()
                title = (md.get('title') or '').strip()
                source = (md.get('source') or '').strip()
                label = url or title or source or 'Unknown source'
                # Normalize PDF vs website hint
                kind = 'PDF' if 'pdf' in (src_type or '').lower() or (label.lower().endswith('.pdf')) else ('Website' if url else 'Document')
                references.append({
                    'index': i,
                    'label': label,
                    'type': kind,
                    'url': url,
                })

            if self.use_openai:
                raw = self._generate_openai_response(user_input, context, references)
            else:
                raw = self._generate_local_response(user_input, context)
            return self._sanitize_output(raw)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    async def generate_response_async(self, user_input: str) -> str:
        """Async version: builds citations and uses async OpenAI client when available."""
        try:
            # Offload embedding/search to a thread with bounded concurrency
            async with EMBEDDING_SEMAPHORE:
                top_results = await asyncio.to_thread(self.knowledge_base.search_similar_content, user_input, 3)

            ctx_parts = []
            for r in top_results:
                c = (r.get('content') or '').strip()
                if c:
                    ctx_parts.append(c[:800])
            context = "\n\n".join(ctx_parts)

            references = []
            for i, r in enumerate(top_results, start=1):
                md = r.get('metadata') or {}
                src_type = md.get('type') or ''
                url = (md.get('url') or '').strip()
                title = (md.get('title') or '').strip()
                source = (md.get('source') or '').strip()
                label = url or title or source or 'Unknown source'
                kind = 'PDF' if 'pdf' in (src_type or '').lower() or (label.lower().endswith('.pdf')) else ('Website' if url else 'Document')
                references.append({'index': i, 'label': label, 'type': kind, 'url': url})

            if self.use_openai:
                if getattr(self, "async_client", None) is not None:
                    raw = await self._generate_openai_response_async(user_input, context, references)
                else:
                    raw = await asyncio.to_thread(self._generate_openai_response, user_input, context, references)
            else:
                raw = await asyncio.to_thread(self._generate_local_response, user_input, context)
            return self._sanitize_output(raw)
        except Exception as e:
            logger.error(f"Error generating async response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def _generate_openai_response(self, user_input: str, context: str, references: list) -> str:
        """Generate response using OpenAI API without citations"""
        try:
            system_prompt = (
                "You are a helpful chatbot. Answer concisely and naturally. "
                "Use the provided Context to ground your answer. "
                "Do not include any citations, reference numbers, or source attributions in your response. "
                "Provide a clean, direct answer based on the context information."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Context:\n{context}"},
                {"role": "user", "content": user_input},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return "I'm having trouble connecting to the AI service."

    async def _generate_openai_response_async(self, user_input: str, context: str, references: list) -> str:
        """Generate response using OpenAI API asynchronously without citations."""
        try:
            if getattr(self, "async_client", None) is None:
                return await asyncio.to_thread(self._generate_openai_response, user_input, context, references)

            system_prompt = (
                "You are a helpful chatbot. Answer concisely and naturally. "
                "Use the provided Context to ground your answer. "
                "Do not include any citations, reference numbers, or source attributions in your response. "
                "Provide a clean, direct answer based on the context information."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Context:\n{context}"},
                {"role": "user", "content": user_input},
            ]

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI API (async): {e}")
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

    # ----------------------------
    # Post-processing utilities
    # ----------------------------
    def _sanitize_output(self, text: str) -> str:
        """Replace the legacy email 'invest@arvocap.com' with 'clients@arvocap.com'.

        This ensures bot responses use the updated email address while preserving
        all other knowledge content unchanged.
        """
        if not text:
            return text
        return self._legacy_email_pattern.sub(self._new_email, text)

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
