import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "models/Qwen/Qwen3-VL-30B-A3B-Instruct"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    # device_map="auto",
    device_map='cuda',
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

def describe(video_path):
    # Messages containing a video url(or a local path) and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    for input in inputs:
        print(input, inputs[input].shape)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, 
        max_new_tokens=2048,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

start = time.time()
describe("asserts/delivery-2.mp4")
duration = time.time() - start
print(f"Time for one inference: {duration:.2f} seconds")

start = time.time()
describe("asserts/delivery-1.mp4")
duration = time.time() - start
print(f"Time for one inference: {duration:.2f} seconds")