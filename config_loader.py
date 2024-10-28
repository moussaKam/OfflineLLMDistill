import argparse
import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Override config parameters")

    # Dataset arguments
    parser.add_argument("--dataset.name_or_path", type=str, help="Path to dataset file")
    parser.add_argument("--dataset.split", type=str, help="Dataset split to use")
    parser.add_argument(
        "--dataset.num_samples", type=int, help="Number of samples to use"
    )

    # Model arguments
    parser.add_argument("--models.student", type=str, help="Student model name")
    parser.add_argument("--models.teacher", type=str, help="Teacher model name")

    # Tokenizer arguments
    parser.add_argument("--tokenizer.max_length", type=int, help="Max token length")

    # Training arguments
    parser.add_argument("--training.output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--training.num_train_epochs", type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--training.per_device_train_batch_size", type=int, help="Batch size per device"
    )
    parser.add_argument(
        "--training.gradient_accumulation_steps",
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--training.save_strategy", type=str, help="Save strategy")
    parser.add_argument("--training.logging_steps", type=int, help="Logging steps")
    parser.add_argument("--training.learning_rate", type=float, help="Learning rate")
    parser.add_argument("--training.weight_decay", type=float, help="Weight decay")
    parser.add_argument("--training.warmup_ratio", type=float, help="Warmup ratio")
    parser.add_argument(
        "--training.lr_scheduler_type", type=str, help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--training.resume_from_checkpoint",
        type=str,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--training.fp16", type=lambda x: x.lower() == "true", help="Use fp16 precision"
    )
    parser.add_argument(
        "--training.bf16", type=lambda x: x.lower() == "true", help="Use bf16 precision"
    )
    parser.add_argument(
        "--training.remove_unused_columns",
        type=lambda x: x.lower() == "true",
        help="Remove unused columns",
    )
    parser.add_argument(
        "--training.gradient_checkpointing",
        type=lambda x: x.lower() == "true",
        help="Use gradient checkpointing",
    )

    # Distillation arguments
    # parser.add_argument("--distillation.temperature", type=float, help="Distillation temperature")
    parser.add_argument("--distillation.alpha", type=float, help="Distillation alpha")
    parser.add_argument("--distillation.top_k", type=int, help="Top K logits")
    parser.add_argument(
        "--distillation.do_generate_inference_outputs",
        type=lambda x: x.lower() == "true",
        help="Generate inference outputs",
    )
    parser.add_argument(
        "--distillation.inference_outputs_dir",
        type=str,
        help="Inference outputs directory",
    )

    # Model config arguments
    parser.add_argument(
        "--model_config.use_flash_attention",
        type=lambda x: x.lower() == "true",
        help="Use flash attention",
    )

    # DeepSpeed arguments
    parser.add_argument(
        "--use_deepspeed", type=lambda x: x.lower() == "true", help="Enable DeepSpeed"
    )
    parser.add_argument(
        "--deepspeed_config_file", type=str, help="Path to DeepSpeed config file"
    )

    args = parser.parse_args()
    return vars(args)


def override_config(config, overrides):
    for key, value in overrides.items():
        if value is not None:
            keys = key.split(".")
            sub_config = config
            for k in keys[:-1]:
                sub_config = sub_config.setdefault(k, {})
            sub_config[keys[-1]] = value
    return config


def load_and_override_config(config_path="config.yaml"):
    config = load_config(config_path)
    overrides = parse_args()
    config = override_config(config, overrides)
    print(config)
    return config
