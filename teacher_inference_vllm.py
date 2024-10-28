import pandas as pd
from vllm import LLM, SamplingParams
import torch
import os
from config_loader import load_and_override_config, parse_args

def get_prompts(input, output=None):
    messages = [
        {"role": "user", "content": input},
    ]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages

args = parse_args()
config_path = args.pop("config_path")
config = load_and_override_config(config_path)

df = pd.read_json(
    config["dataset"]["name_or_path"],
    nrows=config["dataset"]["num_samples"],
    lines=True,
)

llm = LLM(
    model=config["models"]["teacher"],
    tensor_parallel_size=config["vllm"]["tensor_parallel_size"],
    trust_remote_code=True,
    dtype="auto",
    gpu_memory_utilization=config["vllm"]["gpu_memory_utilization"],
    max_logprobs=max(20, config["distillation"]["top_k"]),
    enable_chunked_prefill=True,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=(
        1000 if config["distillation"]["do_generate_inference_outputs"] else 1
    ),  # add argument
    logprobs=config["distillation"]["top_k"],
    prompt_logprobs=config["distillation"]["top_k"],
)

print("Sampling params: ", sampling_params)

if config["distillation"]["do_generate_inference_outputs"]:
    prompts = [get_prompts(el) for el in df["input"].tolist()]
else:
    prompts = [get_prompts(el1, el2) for el1, el2 in zip(df["input"], df["output"])]

print(prompts[0])


outputs = llm.chat(
    prompts,
    sampling_params=sampling_params,
)


prompt_logprobs = [
    output.prompt_logprobs[1:] for output in outputs
]  # check why the first element is None

if config["distillation"]["do_generate_inference_outputs"]:
    generated_logprobs = [output.outputs[0].logprobs for output in outputs]

top_k_logits_and_indices = []

for idx in range(len(prompt_logprobs)):
    example_prompt_logprobs = []
    example_prompt_logprobs_indexes = []
    example_generated_logprobs_indexes = []
    example_generated_logprobs = []

    for prompt_logprob in prompt_logprobs[idx]:
        list_logprob = sorted(
            list(prompt_logprob.items()), key=lambda x: x[1].rank, reverse=False
        )
        if len(list_logprob) > config["distillation"]["top_k"]:
            list_logprob = list_logprob[: config["distillation"]["top_k"]]
        example_prompt_logprobs_indexes.append([el[0] for el in list_logprob])
        example_prompt_logprobs.append([el[1].logprob for el in list_logprob])

    if config["distillation"]["do_generate_inference_outputs"]:
        example_generated_logprobs_indexes = [
            list(logprob.keys()) for logprob in generated_logprobs[idx]
        ]
        example_generated_logprobs = [
            [el.logprob for el in logprob.values()]
            for logprob in generated_logprobs[idx]
        ]

    top_k_logits_and_indices.append(
        {
            "top_k_logprobs": torch.tensor(
                example_prompt_logprobs + example_generated_logprobs
            ),
            "top_k_indexes": torch.tensor(
                example_prompt_logprobs_indexes + example_generated_logprobs_indexes
            ),
        }
    )

    prompts = [output.prompt for output in outputs]
    output_texts = [
        (
            output.outputs[0].text
            if config["distillation"]["do_generate_inference_outputs"]
            else ""
        )
        for output in outputs
    ]

output_file = "top_k_logits_and_indices.pt"

if not os.path.exists(config["distillation"]["inference_outputs_dir"]):
    os.makedirs(config["distillation"]["inference_outputs_dir"])
torch.save(
    top_k_logits_and_indices,
    os.path.join(config["distillation"]["inference_outputs_dir"], output_file),
)
prompts_outputs = pd.DataFrame({"prompts": prompts, "outputs": output_texts})
prompts_outputs.to_json(
    os.path.join(
        config["distillation"]["inference_outputs_dir"], "prompts_outputs.json"
    ),
    orient="records",
    lines=True,
    force_ascii=False,
)
