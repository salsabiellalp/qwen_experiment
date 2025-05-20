
from unsloth import FastLanguageModel
import torch
from transformers import BitsAndBytesConfig
import pandas as pd
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import time
import psutil
import shutil
import torch

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="qwen_baseline")
parser.add_argument("--pretrained_model_path", type=str, default="unsloth/Qwen2-7B")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
parser.add_argument("--use_rslora", action="store_true", help="Use RSLoRA if specified")
parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing if specified")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--hf_token", type=str)

args = parser.parse_args()


# lora_llama
# lora_llama_instruct
# lora_qwen
# lora_qwen_instruct
# lora_sea_v3
# lora_sea_v2.5
model_name = args.model_name
max_seq_length = args.max_seq_length
load_in_4bit = True
dtype = None

quant_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Step 1: Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.pretrained_model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    quantization_config = quant_config,
    use_gradient_checkpointing=args.use_gradient_checkpointing,
)

# Step 2: Apply LoRA with specific rank and alpha
model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    use_rslora=args.use_rslora,
)


df = pd.read_csv("/kaggle/working/train_data_genai.csv")

zsl_prompt = """Berikut adalah sebuah instruksi yang menjelaskan sebuah tugas, diikuti dengan sebuah input yang memberikan konteks tambahan. Tulislah respons yang sesuai untuk menyelesaikan permintaan tersebut.

### Instruction:
Tentukan apakah teks berikut merupakan pesan penipuan, pesan promo, atau pesan normal. Jawab dengan hanya menggunakan satu kata (Penipuan/Promo/Normal).

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = zsl_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# Load dan format dataset kamu sendiri
df = pd.read_csv("train_data_genai.csv")
dataset = Dataset.from_pandas(df)

dataset = dataset.map(formatting_prompts_func, batched=True)


class SaveAtEpochCallback(TrainerCallback):
    def __init__(self, save_epochs, output_dir, hf_repo, hf_token, tokenizer):
        self.save_epochs = save_epochs
        self.output_dir = output_dir
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.tokenizer = tokenizer  # simpan langsung

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch in self.save_epochs:
            epoch_int = int(state.epoch)
            save_path = f"{self.output_dir}/epoch-{epoch_int}"
            print(f">>> Saving model locally to: {save_path}")

            model = kwargs["model"]
            model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            print(f">>> Pushing model to Hugging Face Hub at branch: epoch-{epoch_int}")

            # Push to Hugging Face Hub with specific branch
            model.push_to_hub(
                repo_id=self.hf_repo,
                token=self.hf_token,
                revision=f"epoch-{epoch_int}"
            )

            self.tokenizer.push_to_hub(
                repo_id=self.hf_repo,
                token=self.hf_token,
                revision=f"epoch-{epoch_int}"
            )

        return control

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = args.max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 5, # Set this for 1 full training run.
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_strategy = "epoch",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none" # Use this for WandB etc
    ),
    callbacks=[SaveAtEpochCallback(
        save_epochs=[1, 3, 5],
        output_dir="./saved_models",
        hf_repo=f"ilybawkugo/{model_name}",
        hf_token=args.hf_token,
        tokenizer=tokenizer)]
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Monitoring start
start_time = time.time()
process = psutil.Process()
gpu_props = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_gpu_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU: {gpu_stats.name} | Max memory: {max_gpu_memory} GB")
print(f"GPU memory reserved before training: {start_gpu_memory} GB")

trainer_stats = trainer.train()

def push_model(model, tokenizer, output_dir, hf_repo, hf_token, branch="epoch-5"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.push_to_hub(repo_id=hf_repo, token=hf_token, revision=branch)
    tokenizer.push_to_hub(repo_id=hf_repo, token=hf_token, revision=branch)

hf_repo=f"ilybawkugo/{model_name}"
hf_token=args.hf_token

push_model(model, tokenizer, model_name, hf_repo, hf_token)

# Monitoring end
end_time = time.time()
duration = round(end_time - start_time, 2)
ram_used = round(process.memory_info().rss / 1024**2, 2)
cpu_percent = psutil.cpu_percent(interval=1)
gpu_mem_peak = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
gpu_used = round(gpu_mem_peak - start_gpu_memory, 2)

disk = shutil.disk_usage("/")
disk_used_gb = round((disk.total - disk.free) / 1024**3, 2)
disk_free_gb = round(disk.free / 1024**3, 2)

print(f"Runtime: {duration}s")
print(f"RAM used: {ram_used} MB | CPU: {cpu_percent}%")
print(f"GPU mem used: {gpu_used} MB (of {gpu_mem_peak} MB peak)")
print(f"Disk used: {disk_used_gb} GB | Free: {disk_free_gb} GB")

# Additional GPU memory usage details (using torch)
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds ({round(trainer_stats.metrics['train_runtime']/60, 2)} minutes)")
print(f"Peak reserved GPU memory: {used_memory} GB ({used_percentage}% of max)")
print(f"Reserved GPU memory for training: {used_memory_for_lora} GB ({lora_percentage}% of max)")

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")