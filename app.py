import gradio as gr
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载微调的 Whisper 模型
asr_pipeline = pipeline("automatic-speech-recognition", model="/root/autodl-fs/whisper-large-v3-turbo-continue", device=device)

# 加载微调的 Phi-3 Mini 模型
correction_pipeline = pipeline("text2text-generation", model="/root/autodl-fs/phi3-mini_finetune", max_new_tokens=512, device=device)

def asr_correction_pipeline(audio_file):
    """
    音频转文本 + 文本纠错流程
    :param audio_file: 上传的音频文件路径
    :return: 转录结果和纠错结果
    """
    # Step 1: ASR 转录
    asr_result = asr_pipeline(audio_file, generate_kwargs={"language": "zh", "task": "transcribe"})["text"]

    # Step 2: 文本纠错
    prompt = f"你是一名语音识别纠错助手。你的任务是根据用户的语音识别结果，纠正其中的错误。如果语音识别结果正确，请保持不变。\n语音识别结果: {asr_result}\n纠正后的文本: "
    corrected_text = correction_pipeline(prompt)[0]["generated_text"]

    return asr_result, corrected_text

# 使用 Gradio 构建界面
def build_interface():
    # Gradio 的输入和输出组件
    audio_input = gr.Audio(type="filepath", label="上传或录制音频")
    asr_output = gr.Textbox(label="初步转录结果")
    corrected_output = gr.Textbox(label="纠错后文本")

    # 接口绑定函数
    interface = gr.Interface(
        fn=asr_correction_pipeline,
        inputs=audio_input,
        outputs=[asr_output, corrected_output],
        title="ASR 系统",
        description="使用 Whisper 模型和 Phi-3 Mini 模型实现语音识别与纠错。",
    )
    return interface

# 启动 Gradio 界面
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=True)

