# 齿问大模型 DentalGPT

<div align="center">
<h3>
  DentalGPT
</h3>
</div>

<div align="center">
<h4>
 🤖 <a href="https://huggingface.co/Eric3200/DentalGPT-7B-1026" target="_blank">Dental-7B-1026</a>
</h4>
</div>

## ⚡ Introduction
Hello! Welcome to the repository for DentalGPT (齿问大模型)! You can access DentalGPT-7B-1026 on [HF space](https://huggingface.co/spaces/Eric3200/DentalGPT).

**DentalGPT** is a specialized medical LLM for dental image analysis. Through a customized pretraining and SFT, we utilized **more than 100,000 dental images** to enhance the performance of Qwen2.5-VL-7B-Instruct in this specific domain, achieving a level comparable to models such as Claude-Sonnet-4.5-Thinking, GPT-5, and Gemini2.5-Pro.

## 👨‍⚕️ Model

#### Model Access

> **DentalGPT-7B-Preview** is available on Huggingface:

|                        | Parameters |  Link                                                                  |
| ---------------------- | ---------- | --------------------------------------------------------------------- |
| **DentalGPT-7B-1026**  | 7B         | [Huggingface Link](https://huggingface.co/Eric3200/DentalGPT-7B-1026) |

*Note: DentalGPT-7B-1026 is based on Qwen2.5-VL-7B-Instruct.*

#### Model Inference

<details open>
<summary><h4>Inference with DentalGPT</h4></summary>

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("Eric3200/DentalGPT-7B-1026")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Eric3200/DentalGPT-7B-1026", torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/path/to/your/image.png",
            },
            {"type": "text", "text": "Please analyze this image."},
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

<details close>
<summary><h4>Evaluation Settings</h></summary>

#### Evaluation Data Collection
To evaluate the capability of multimodal large language models (MLLMs) in understanding dental images, we curated a specialized evaluation dataset sourced from hospital and internet data.

The dataset is composed of three subsets:
1. **Intraoral-Classification-I-270**
- A collection of intraoral photographs from the [AlphaDent](https://www.kaggle.com/competitions/alpha-dent) dataset, captured by licensed dentists from a clinical perspective under standardized lighting and imaging conditions.
- These images provide high-quality professional references of oral health conditions.
- Included labels: *Tooth discoloration, Abnormal gingival coloration, Gingival recession, Dental caries, Tooth pigmentation, Tooth defect or loss, Tooth loss, Dental calculus, Abnormal tooth morphology, Abnormal gingival morphology.*
2. **Intraoral-Classification-II-207**
- A set of intraoral images collected from the internet based on dental-related keywords.
– The images feature diverse lighting and shooting angles, simulating photos that patients might take themselves.
- Included labels: *Tooth pigmentation, Abnormal gingival coloration, Dental calculus, Tooth loss, Dental caries, Abnormal gingival morphology, Gingival recession.*
3. **Panorama-Classification-156**
- A dataset of panoramic dental radiographs (X-rays) provided by Shenzhen Stomatology Hospital (Pingshan) of Southern Medical University, containing real patient panoramic imaging data.
- Included labels: *Periodontal disease, Root canal treatment, Tooth defect or loss, Jawbone lesion, Periapical lesion, Impacted tooth.*
Together, these subsets cover both clinical and in-the-wild dental imaging conditions, ensuring a comprehensive evaluation of the models’ visual diagnostic abilities.

#### Annotation and Filtering
All images were **annotated by professional dentists** from Shenzhen Stomatology Hospital (Pingshan) of Southern Medical University.

To ensure high data reliability, only samples with an **inter-annotator agreement above 90% were retained for evaluation**.

Additionally, **all labels were balanced** between positive and negative samples to ensure fairness across categories.
This enables direct comparison of model accuracy across different disease types.

#### Evaluation Protocol
Each evaluated MLLM was required to determine whether a given image indicates the presence (“Yes”) or absence (“No”) of a specific dental condition.

The model’s final output must be strictly binary (“Yes” or “No”), without providing explanations. (Note: Reasoning-enabled models may internally generate their chain of thought.)
</details>
<details open>
<summary><h4>Evaluation Results</h4></summary>
  
|      | Intraoral-Classification-I-270 | Intraoral-Classification-II-207 | Panorama-Classification-156 | Average |
| ----- | ----- | ----- | ----- | ----- |
| Claude-Sonnet-4.5-Thinking | 55.2 | 66.7 | 55.8 | 59.2 |
| Qwen3-VL-235B-A22B-Thinking | 56.7 | 65.7 | 55.8 | 59.4 |
| Gemini-2.5-Pro | 57.0 | 65.2 | 63.5 | 61.9 |
| GPT-5 | 59.3 | 71.0 | 63.5 | 64.6 |
| Qwen2.5-VL-7B-Instruct | 54.8 | 61.8 | 50.0 | 55.5 |
| **DentalGPT-7B-1026** | **63.2** | **75.8** | **80.1** | **73.0** |

> 'Intraoral' denotes intraoral photographs, while 'Panorama' denotes panoramic radiographs. All images were annotated by licensed dentists.
</details>

## 🎯 To-Do
- [x] Commit the preview version of DentalGPT
- [ ] Upload the technical report
- [ ] Propose the paper
- [ ] Release the professional version of DentalGPT

##  📖 About Us
We are from:
- The Chinese University of Hong Kong, Shenzhen 香港中文大学（深圳）
- Shenzhen Stomatology Hospital (Pingshan) of Southern Medical University 南方医科大学深圳口腔医院（坪山）
- Faculty of Dentistry, The University of Hong Kong 港大牙医学院
- Freedom AI 深圳自由动脉科技有限公司

特别鸣谢智谱华章科技提供支持。
