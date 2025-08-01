#!/usr/bin/env python3
"""
GPT-2 Inference Script

This script provides inference capabilities for the GPT-2 model from Hugging Face.
GPT-2 is a robust language model suitable for high-quality text generation tasks.

Author: Maxx Richard Rahman
Model: https://huggingface.co/gpt2
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
import sys
import numpy as np
from typing import Optional, List


class GPT2Inference:
    """
    A class to handle inference with the GPT-2 model.
    """
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the GPT-2 inference engine.
        
        Args:
            model_name (str): The Hugging Face model identifier
            device (str, optional): Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        print(f"Initializing GPT-2 on device: {self.device}")
        self.load_model()
    
    def _get_best_device(self) -> str:
        """
        Automatically select the best available device.
        
        Returns:
            str: The best available device
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """
        Load the tokenizer and model from Hugging Face.
        """
        try:
            print(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad_token if it doesn't exist (common for GPT-2)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Loading model from {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Create a text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def generate_text(self, 
                     prompt: str, 
                     max_new_tokens: int = 80, 
                     temperature: float = 0.7, 
                     top_p: float = 0.9,
                     num_return_sequences: int = 1,
                     do_sample: bool = True) -> List[str]:
        """
        Generate text using the GPT-2 model.
        
        Args:
            prompt (str): Input text prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
            top_p (float): Top-p (nucleus) sampling parameter
            num_return_sequences (int): Number of sequences to generate
            do_sample (bool): Whether to use sampling
            
        Returns:
            List[str]: Generated text sequences
        """
        try:
            print(f"Generating text for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Improve the prompt for better generation
            improved_prompt = self.improve_prompt(prompt)
            
            # Generate text using the pipeline
            outputs = self.pipeline(
                improved_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=1.2,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample
            )
            
            # Extract generated text
            generated_texts = []
            for output in outputs:
                generated_text = output['generated_text']
                # Remove the original prompt from the output
                if generated_text.startswith(improved_prompt):
                    generated_text = generated_text[len(improved_prompt):].strip()
                elif generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                # Clean up the generated text
                generated_text = self._clean_generated_text(generated_text)
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            print(f"Error during text generation: {e}")
            return []
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean up generated text to improve readability.
        
        Args:
            text (str): Raw generated text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace and newlines
        text = ' '.join(text.split())
        
        # For GPT-2, let's be less aggressive about cutting off text
        # Remove only obvious artifacts and repetitive characters
        
        # Remove excessive repetitive punctuation
        import re
        text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)  # Remove excessive repeated punctuation
        text = re.sub(r'\s{2,}', ' ', text)  # Remove multiple spaces
        
        # Stop at natural sentence boundaries, but be less restrictive
        sentences = text.split('.')
        if len(sentences) > 1:
            # Find first substantial sentence
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 15:  # Lowered threshold
                    return sentence.strip() + '.'
        
        # If no good sentence break, look for other punctuation
        for punct in ['!', '?']:
            if punct in text:
                parts = text.split(punct)
                if len(parts[0].strip()) > 15:
                    return parts[0].strip() + punct
        
        # Return first reasonable chunk if no punctuation found
        words = text.split()
        if len(words) > 15:
            return ' '.join(words[:15]) + '...'
        
        return text.strip()
    
    def improve_prompt(self, prompt: str) -> str:
        """
        Improve the input prompt for better generation with GPT-2.
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Improved prompt
        """
        # GPT-2 works well with most prompts as-is, just ensure proper formatting
        prompt = prompt.strip()
        
        # For questions, keep them as questions
        if prompt.endswith('?'):
            return prompt
        
        # For incomplete sentences, don't add punctuation as GPT-2 handles continuation well
        return prompt
    
    def interactive_mode(self):
        """
        Run the model in interactive mode for continuous text generation.
        """
        print("\n" + "="*50)
        print("GPT-2 Interactive Mode")
        print("="*50)
        print("Enter your prompts below. Type 'quit' or 'exit' to stop.")
        print("Type 'settings' to modify generation parameters.")
        print("="*50 + "\n")
        
        # Default settings
        settings = {
            'max_new_tokens': 80,
            'temperature': 0.7,
            'top_p': 0.9,
            'num_sequences': 1
        }
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'settings':
                    self._modify_settings(settings)
                    continue
                
                if not user_input:
                    print("Please enter a prompt.")
                    continue
                
                # Generate text
                generated_texts = self.generate_text(
                    user_input,
                    max_new_tokens=settings['max_new_tokens'],
                    temperature=settings['temperature'],
                    top_p=settings['top_p'],
                    num_return_sequences=settings['num_sequences']
                )
                
                # Display results
                print(f"\nGenerated text:")
                print("-" * 30)
                for i, text in enumerate(generated_texts, 1):
                    if settings['num_sequences'] > 1:
                        print(f"Sequence {i}: {text}")
                    else:
                        print(text)
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _modify_settings(self, settings: dict):
        """
        Allow user to modify generation settings.
        
        Args:
            settings (dict): Current settings dictionary
        """
        print("\nCurrent settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("\nEnter new values (press Enter to keep current value):")
        
        try:
            new_max_new_tokens = input(f"Max new tokens ({settings['max_new_tokens']}): ").strip()
            if new_max_new_tokens:
                settings['max_new_tokens'] = int(new_max_new_tokens)
            
            new_temperature = input(f"Temperature ({settings['temperature']}): ").strip()
            if new_temperature:
                settings['temperature'] = float(new_temperature)
            
            new_top_p = input(f"Top-p ({settings['top_p']}): ").strip()
            if new_top_p:
                settings['top_p'] = float(new_top_p)
            
            new_num_sequences = input(f"Number of sequences ({settings['num_sequences']}): ").strip()
            if new_num_sequences:
                settings['num_sequences'] = int(new_num_sequences)
            
            print("Settings updated!")
            
        except ValueError as e:
            print(f"Invalid input: {e}")


def main():
    """
    Main function to handle command line interface.
    """
    parser = argparse.ArgumentParser(description="GPT-2 Inference Script")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num-sequences", type=int, default=1, help="Number of sequences to generate")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'mps'], help="Device to use for inference")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the inference engine
    llm = GPT2Inference(device=args.device)
    
    if args.interactive:
        # Run in interactive mode
        llm.interactive_mode()
    elif args.prompt:
        # Single prompt mode
        generated_texts = llm.generate_text(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        print(f"\nPrompt: {args.prompt}")
        print("Generated text:")
        print("-" * 50)
        for i, text in enumerate(generated_texts, 1):
            if args.num_sequences > 1:
                print(f"Sequence {i}: {text}")
            else:
                print(text)
        print("-" * 50)
    else:
        # Default: show help and run interactive mode
        parser.print_help()
        print("\nNo prompt provided. Starting interactive mode...\n")
        llm.interactive_mode()


if __name__ == "__main__":
    main()
