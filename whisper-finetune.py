# 1. 导入必要的库
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# 2. 数据加载
common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", 
    "zh-CN", 
    split="train+validation", 
    trust_remote_code=True
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", 
    "zh-CN", 
    split="test", 
    trust_remote_code=True
)

# 移除不必要的列
columns_to_remove = [
    "accent", "age", "client_id", "down_votes", 
    "gender", "locale", "path", "segment", "up_votes"
]
common_voice = common_voice.remove_columns(columns_to_remove)

# 3. 初始化处理器和模型
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-base", 
    language="Chinese", 
    task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", 
    language="Chinese", 
    task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# 设置模型配置
model.generation_config.language = "chinese"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# 4. 数据预处理
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset, 
    remove_columns=common_voice.column_names["train"], 
    num_proc=4
)

# 5. 数据整理器
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 6. 评估指标
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 7. 训练配置
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-cn",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# 8. 训练器设置和训练
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# 9. 开始训练
trainer.train()