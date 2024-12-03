import sys
import logging

import datasets
from datasets import load_dataset, Dataset
# from peft import LoraConfig
import torch
import transformers
import wandb
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

from utils import read_json, computer_metrics

"""
A simple example on using SFTTrainer and Accelerate to finetune Phi-3 models. For
a more advanced example, please follow HF alignment-handbook/scripts/run_sft.py.
This example has utilized DeepSpeed ZeRO3 offload to reduce the memory usage. The
script can be run on V100 or later generation GPUs. Here are some suggestions on 
futher reducing memory consumption:
    - reduce batch size
    - decrease lora dimension
    - restrict lora target modules
Please follow these steps to run the script:
1. Install dependencies: 
    conda install -c conda-forge accelerate
    pip3 install -i https://pypi.org/simple/ bitsandbytes
    pip3 install peft transformers trl datasets
    pip3 install deepspeed
2. Setup accelerate and deepspeed config based on the machine used:
    accelerate config
Here is a sample config for deepspeed zero3:
    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
      gradient_accumulation_steps: 1
      offload_optimizer_device: none
      offload_param_device: none
      zero3_init_flag: true
      zero3_save_16bit_model: true
      zero_stage: 3
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
3. check accelerate config:
    accelerate env
4. Run the code:
    accelerate launch sample_finetune.py
"""

logger = logging.getLogger(__name__)


###################
# Hyper-parameters
###################
training_config = {
    "bf16": True,
    "do_eval": True,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "max_steps": 300,
    "output_dir": "/root/autodl-fs/phi3-mini_checkpoint",
    "overwrite_output_dir": True,
    "eval_strategy": "steps",
    "eval_steps": 10,
    "per_device_eval_batch_size": 8,
    "per_device_train_batch_size": 32,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "report_to": "wandb",
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
    }

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}
train_conf = TrainingArguments(**training_config)
# peft_conf = LoraConfig(**peft_config)


###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
# logger.info(f"PEFT parameters {peft_conf}")


################
# Model Loading
################
# checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
checkpoint_path = "/root/autodl-fs/phi3-mini_checkpoint/checkpoint-300"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


##################
# Data Processing
##################
data = read_json("/root/autodl-fs/data/whisper_label_result.json")

raw_dataset = Dataset.from_list(data)

# Split into train and test datasets
train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

def preprocess_data_with_chinese_prompt(example):
    """
    Preprocess the input example to include a loss mask for the label part,
    using the new Chinese prompt.
    """
    # Extract ASR input and the correct label
    asr_input = example["whisper"]
    label = example["label"]
    
    # Define the Chinese prompt
    input_text = f"你是一名语音识别纠错助手。你的任务是根据用户的语音识别结果，纠正其中的错误。如果语音识别结果正确，请保持不变。\n语音识别结果: {asr_input}\n纠正后的文本: "
    full_text = input_text + label

    # Tokenize the full text
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=2048)
    
    # Create labels with a loss mask
    label_start = len(tokenizer(input_text)["input_ids"])  # The start index of the label part
    labels = [-100] * label_start + tokenized["input_ids"][label_start:]
    labels = labels[:2048]  # Ensure labels are the same length as the input

    tokenized["labels"] = labels
    return tokenized

# Preprocess datasets with the new function
processed_train_dataset = train_dataset.map(
    preprocess_data_with_chinese_prompt,
    remove_columns=train_dataset.column_names,
    desc="Processing training data with Chinese prompt",
)

processed_test_dataset = test_dataset.map(
    preprocess_data_with_chinese_prompt,
    remove_columns=test_dataset.column_names,
    desc="Processing test data with Chinese prompt",
)


###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
    compute_metrics=computer_metrics
)
train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
# metrics = train_result.metrics
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()


# #############
# # Evaluation
# #############
# tokenizer.padding_side = 'left'
# metrics = trainer.evaluate()
# metrics["eval_samples"] = len(processed_test_dataset)
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model("/root/autodl-fs/phi3-mini_finetune")
tokenizer.save_pretrained("/root/autodl-fs/phi3-mini_finetune")