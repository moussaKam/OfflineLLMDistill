import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.config_loader import load_and_override_config, parse_args
import os
from torch.utils.data import DataLoader

def main():
    args = parse_args()
    config_path = args.pop("config_path")
    config = load_and_override_config(config_path)

    def get_prompts(input, output, tokenizer):
        messages = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": output},
        ]
        chat_format = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"text": chat_format}


    dataset = load_dataset("json", data_files=config["dataset"]["name_or_path"], split="train")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])

    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda x: get_prompts(x["input"], x["output"], teacher_tokenizer),
        remove_columns=original_columns,
        num_proc=8,
    )


    def tokenize_function(examples):
        return teacher_tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="longest",
        )


    tokenized_dataset = dataset.map(
        tokenize_function,
        num_proc=8,
        remove_columns=["text"],
        batched=True,
        batch_size=32,
    )
    
    tokenized_dataset.set_format(type="torch")

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"], device_map="auto", **model_kwargs
    )
    data_loader = DataLoader(tokenized_dataset, batch_size=4, shuffle=False)
    top_k_logits_and_indices = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating teacher logits"):
            batch["input_ids"] = batch["input_ids"].to(teacher_model.device)
            batch["attention_mask"] = batch["attention_mask"].to(teacher_model.device)
            
            outputs = teacher_model(**batch)
            logits = outputs.logits
            
            # Get the top K logits and their indices
            # top_k_logits, top_k_indices = torch.topk(logits, config["distillation"]["top_k"], dim=-1)
            # top_k_logits_and_indices.append(
            #     {
            #         "top_k_logits": top_k_logits.cpu(),
            #         "top_k_indices": top_k_indices.cpu(),
            #     }
            # )


    output_file = "top_k_logits_and_indices.pt"

    if not os.path.exists(config["distillation"]["inference_outputs_dir"]):
        os.makedirs(config["distillation"]["inference_outputs_dir"])
    torch.save(
        top_k_logits_and_indices,
        os.path.join(config["distillation"]["inference_outputs_dir"], output_file),
    )

if __name__ == "__main__":
    main()