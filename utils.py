import json
import evaluate


def write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def read_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
        return data


metric = evaluate.load("/root/wer.py")

def computer_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}