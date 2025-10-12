# 齿问大模型 DentalGPT

<div align="center">
<h3>
  DentalGPT
</h3>
</div>

<div align="center">
<h4>
 🤖 <a href="https://www.modelscope.cn/models/Eric3200C/DentalGPT-7B-Preview" target="_blank">Dental-7B-Preview</a>
</h4>
</div>

## ⚡ Introduction
Hello! Welcome to the repository for DentalGPT (齿问大模型)!

**DentalGPT** is a specialized medical LLM for intraoral photograph analysis. Through a customized curriculum learning design and DAPO training, we utilized **1,585 annotated intraoral photographs** to enhance the performance of Qwen2.5-VL-7B-Instruct in this specific domain, achieving a level comparable to models such as GPT-4o, GPT-5, and Gemini2.5-Pro.

> *⚠️ DentalGPT-7B-Preview is an early-stage model for investigating training approaches. It is not equipped with professional diagnostic competence, and any clinical application is strictly discouraged.* 

## 👨‍⚕️ Model

#### Model Access

> **DentalGPT-7B-Preview** is available on Huggingface:

|                        | Parameters |  Link                                                                  |
| ---------------------- | ---------- | --------------------------------------------------------------------- |
| **DentalGPT-7B-Preview**  | 7B         | [ModelScope Link](https://www.modelscope.cn/models/Eric3200C/DentalGPT-7B-Preview) |
| **DentalGPT-9B** | 9B       |  Available soon  |

*Note: DentalGPT-7B-Preview is based on Qwen2.5-VL-7B-Instruct.*

#### Model Inference

<details open>
<summary><h4>Inference with DentalGPT</h4></summary>

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("FreedomIntelligence/DentalGPT-7B-Preview")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("FreedomIntelligence/DentalGPT-7B-Preview", torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/path/to/your/image.png",
            },
            {"type": "text", "text": "请帮我分析这张患者的口内拍摄图片。"},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
</details>

## 🧐 Evaluation

<details open>
<summary><h4>Out-of-domain Evaluation on 18 oral disease lables</h4></summary>

In this study, we trained DentalGPT-preview on web-crawled dental images, each carefully annotated by experienced dental professionals into 18 categories.

We conduct the out-of-domain evaluation on [AlphaDent](https://www.kaggle.com/competitions/alpha-dent) dataset's test set, which was annotated with the same 18 categories by doctors, ensuring label consistency to reliably evaluate the model’s generalization across different data sources.

|      | Macro F1 | Micro F1 | Macro AUC | Micro AUC | Jaccard Index |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Gemini-2.5-Pro | 42.61 | 42.86 | 56.62 | 55.85 | 47.68 |
| Doubao-1.5-Vision-Pro | 32.80 | 41.33 | 59.83 | 61.10 | 60.55 |
| GPT-4o | 41.16 | 45.03 | 59.67 | 60.55 | 60.75 |
| GPT-5 | **45.58** | 46.89 | 60.44 | 62.28 | 62.25 |
| Qwen2.5-VL-7B-Instruct | 33.16 | 39.28 | 53.57 | 55.50 | 56.80 |
| DentalGPT-7B-Preview | 43.75 | **47.98** | **63.98** | **63.67** | **64.13** |
</details>

## 🎯 To-Do
- [x] Commit the preview version of DentalGPT
- [x] Upload the technical report
- [ ] Open-source the training code
- [ ] Propose the paper
- [ ] Release the professional version of DentalGPT

##  📖 About Us
We are from:
- The Chinese University of Hong Kong, Shenzhen 香港中文大学（深圳）
- Shenzhen Stomatology Hospital (Pingshan) of Southern Medical University 南方医科大学深圳口腔医院（坪山）
- Faculty of Dentistry, The University of Hong Kong 港大学牙学院
- Freedom AI 深圳自由动脉科技有限公司

特别鸣谢智谱华章科技提供支持。
