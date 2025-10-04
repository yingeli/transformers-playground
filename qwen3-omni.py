import time
import torch
import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor
# from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# Prefer local weights to avoid network
MODEL_PATH = "models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model.disable_talker()

# processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = False

def describe(video_path):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe the video."}
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    for input in inputs:
        print(input, inputs[input].shape)

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(
        **inputs,
        # speaker="Ethan",
        # thinker_return_dict_in_generate=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        return_dict_in_generate=True,
    )

    # Robustly handle both dict-like and tensor outputs
    sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
    text = processor.batch_decode(sequences[:, inputs["input_ids"].shape[1] :],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    print(text)
    if audio is not None:
        sf.write(
            "output.wav",
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

describe("asserts/delivery-2.mp4")

start = time.time()
describe("asserts/delivery-1.mp4")

duration = time.time() - start
print(f"Time for one inference: {duration:.2f} seconds")