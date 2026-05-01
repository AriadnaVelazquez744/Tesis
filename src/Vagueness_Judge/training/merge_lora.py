#!/usr/bin/env python3
"""Merge LoRA adapter with base model to create a single model file for LM Studio."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to base model directory")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                        help="Path to LoRA adapter directory (output from training)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--bf16", action="store_true",
                        help="Save in bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Save in float16 precision")
    
    args = parser.parse_args()
    
    print(f"Loading base model from: {args.base_model_path}")
    
    # Determine dtype
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
    
    # Load LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    
    print("Merging adapter weights with base model...")
    
    # Merge and unload (apply LoRA weights to base model)
    model = model.merge_and_unload()
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_dir}")
    
    # Save merged model and tokenizer
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"Done! Merged model saved to: {output_dir}")
    print(f"Model size should be around 6GB for 3B models, 14GB for 7B models")
    print(f"You can now load this in LM Studio directly from: {output_dir}")


if __name__ == "__main__":
    main()
