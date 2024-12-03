from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import json

def write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)

ds = load_from_disk("/root/autodl-tmp/chinese_asr_sample")
model_id = "/root/autodl-fs/whisper-large-v3-turbo-continue"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device
)

result = pipe(ds['audio'], generate_kwargs={"language": "zh", "task": "transcribe"})

write_json("/root/autodl-fs/data/whisper_result.json", result)

res = []

for r, m in zip(result, ds['messages']):
    res.append({"whisper": r['text'], "label": m[1]["content"]})

write_json("/root/autodl-fs/data/whisper_label_result.json", res)


