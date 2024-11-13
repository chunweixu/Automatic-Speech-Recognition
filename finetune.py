import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset

# 1. 加载Whisper模型和处理器
model_name = "openai/whisper-base"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# 2. 加载数据集
# 使用自己的数据集进行微调，可以加载自定义数据集或使用Hugging Face datasets中的公开数据集
# 假设数据集包含 'audio' 和 'transcription' 字段
# dataset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="train[:1%]", trust_remote_code=True)
train_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="train", trust_remote_code=True)
eval_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="validation", trust_remote_code=True)
test_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="test", trust_remote_code=True)

# 3. 数据预处理函数
# def prepare_dataset(batch):
#     audio = batch["audio"]
#     # 将音频转换为模型所需的输入格式
#     batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
#     # 将转录文本转换为目标词汇表索引
#     batch["labels"] = processor.tokenizer(batch["transcription"], add_special_tokens=True).input_ids
#     return batch
def prepare_dataset(batch):
    audio = batch["audio"]
    # 重采样到16kHz
    if audio["sampling_rate"] != 16000:
        import librosa
        audio_array = librosa.resample(
            y=audio["array"],
            orig_sr=audio["sampling_rate"],
            target_sr=16000
        )
    else:
        audio_array = audio["array"]
    
    # 直接使用 processor 提取特征
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="np"
    )
    batch["input_values"] = inputs["input_values"][0]  # 直接访问字典中的 'input_values'

    # 将转录文本转换为目标词汇表索引
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"], add_special_tokens=True).input_ids

    return batch

# 4. 应用数据预处理
# dataset = dataset.map(prepare_dataset)

# 5. 数据集分割
# 通常需要将数据集分为训练集和验证集，这里简单地进行80/20划分
# train_dataset = dataset.shuffle(seed=42).select(range(int(0.8 * len(dataset))))
# eval_dataset = dataset.shuffle(seed=42).select(range(int(0.8 * len(dataset)), len(dataset)))
train_dataset = train_dataset.map(prepare_dataset)
eval_dataset = eval_dataset.map(prepare_dataset)
test_dataset = test_dataset.map(prepare_dataset)

# 6. 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",           # 输出模型保存目录
    per_device_train_batch_size=16,              # 每个设备的批次大小
    per_device_eval_batch_size=16,               # 每个设备的评估批次大小
    evaluation_strategy="epoch",               # 每个epoch进行验证
    logging_dir="./logs",                       # 日志保存目录
    logging_steps=10,                            # 日志记录间隔步数
    learning_rate=5e-5,                          # 学习率
    num_train_epochs=3,                          # 训练的epoch数量
    save_steps=500,                              # 保存模型的间隔步数
    save_total_limit=2,                          # 只保存最近的2个模型检查点
    fp16=True                                    # 如果设备支持则使用混合精度训练
)

# 7. 定义评价指标
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    # 计算Word Error Rate (WER)
    wer = sum([int(p != l) for p, l in zip(pred_str, label_str)]) / len(label_str)
    return {"wer": wer}

# 8. 初始化Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# 9. 开始训练
trainer.train()
