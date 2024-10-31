import torch
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from kd_losses import ForwardKLWithChunkedOutputLoss

kl_loss_fn = ForwardKLWithChunkedOutputLoss(ignore_index=-100)
import os

from config_loader import load_and_override_config, parse_args

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}

args = parse_args()
config_path = args.pop("config_path")
config = load_and_override_config(config_path)

accelerator = Accelerator()
device = accelerator.device

dataset = load_dataset(
    "json",
    data_files=os.path.join(
        config["distillation"]["inference_outputs_dir"], "prompts_outputs.json"
    ),
    split="train",
)
if "num_samples" in config["dataset"]:
    # dataset["train"] = dataset["train"].select(range(config["dataset"]["num_samples"]))
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])


def get_text(example):
    return {"text": example["prompts"] + example["outputs"] + "<|im_end|>"}


# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(get_text, remove_columns=original_columns, num_proc=1)


def tokenize_function(examples):
    return student_tokenizer(
        examples["text"],
        truncation=True,
        max_length=config["tokenizer"]["max_length"],
        padding="max_length",
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, num_proc=8, remove_columns=["text"]
)

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": dtype_map[config["models"]["dtype"]]}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

student_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["student"], **model_kwargs
)


def add_logits_to_example(example, index):
    # Retrieve values directly to reduce repeated lookups
    top_k = config["distillation"]["top_k"]
    max_length = config["tokenizer"]["max_length"]
    dtype = dtype_map[config["models"]["dtype"]]

    # Retrieve and apply in-place exponential to avoid intermediate allocations
    top_k_probs = saved_logits[index]["top_k_logprobs"][:, :top_k].exp_()
    top_k_indices = saved_logits[index]["top_k_indexes"][:, :top_k]

    original_length = top_k_probs.shape[0]

    # Create the padded tensors only if padding is needed
    if original_length < max_length:
        padded_probs = torch.zeros(max_length, top_k, dtype=dtype)
        padded_indices = torch.zeros(max_length, top_k, dtype=torch.int64)

        # Fill in only the valid range
        padded_probs[:original_length, :] = top_k_probs
        padded_indices[:original_length, :] = top_k_indices
    else:
        # No padding required, keep original size
        padded_probs = top_k_probs
        padded_indices = top_k_indices

    # Assign the padded (or original) values to the example
    example["top_k_probs"] = padded_probs
    example["top_k_indices"] = padded_indices

    return example


if config["distillation"]["alpha"] > 0:
    saved_logits = torch.load(
        os.path.join(
            config["distillation"]["inference_outputs_dir"],
            "top_k_logits_and_indices.pt",
        )
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda example, idx: add_logits_to_example(example, idx),
        with_indices=True,
        num_proc=20,
    )



class LogitsTrainer(SFTTrainer):
    def compute_loss(self, model, inputs):
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        if config["distillation"]["alpha"] == 0:
            return student_outputs.loss

        top_k_probs = inputs.get("top_k_probs", None)
        top_k_indices = inputs.get("top_k_indices", None)

        teacher_probs = torch.zeros(
            student_outputs.logits.shape,
            dtype=dtype_map[config["models"]["dtype"]],
            requires_grad=False,
            device=model.device,
        )

        teacher_probs.scatter_(2, top_k_indices, top_k_probs)

        n_chunks = kl_loss_fn.num_output_chunks

        if config["distillation"]["alpha"] == 1:
            loss = kl_loss_fn(
                student_outputs.logits.chunk(n_chunks, dim=1),
                teacher_probs.chunk(n_chunks, dim=1),
                inputs["labels"],
            )
        else:
            loss = kl_loss_fn(
                student_outputs.logits.chunk(n_chunks, dim=1),
                teacher_probs.chunk(n_chunks, dim=1),
                inputs["labels"],
            ) * config["distillation"]["alpha"] + student_outputs.loss * (
                1 - config["distillation"]["alpha"]
            )

        return loss


# Training arguments
training_arguments = TrainingArguments(
    bf16=config["models"]["dtype"] == "bfloat16",
    fp16=config["models"]["dtype"] == "float16",
    **config["training"],
)

if config["train_on_completion_only"]:
    response_template = config["response_template"]
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=student_tokenizer
    )

# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset,
    tokenizer=student_tokenizer,
    args=training_arguments,
    max_seq_length=config["tokenizer"]["max_length"],
    data_collator=collator if config["train_on_completion_only"] else None,
)

trainer = accelerator.prepare(trainer)

trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

trainer.save_model(config["training"]["output_dir"])

print("Training complete. Model saved to", config["training"]["output_dir"])
