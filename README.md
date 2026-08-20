# 🤖 AI Coding Assistant with Qwen2.5 & Gradio

An open-source, multi-turn conversational AI assistant built using the **Qwen2.5-1.5B-Instruct** large language model, PyTorch, Hugging Face Transformers, and a web interface powered by Gradio.

---

## 🌟 Key Features

* **Multi-Turn Memory:** Maintains conversation history and context across user prompts.
* **Optimized for Free Colab Tier:** Loads weights in 16-bit precision (`float16`) to fit comfortably within NVIDIA T4 GPU memory.
* **Interactive Web Interface:** Clean UI built with Gradio, complete with example prompts, clear buttons, and public shareable URLs.
* **Strict History Parsing:** Handles dynamic conversation state formatting to avoid tokenization bottlenecks.

---

## 🛠️ Tech Stack

* **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Deep Learning Framework:** PyTorch & Hugging Face Transformers
* **Web UI:** Gradio
* **Compute Environment:** Google Colab (T4 GPU) / Local CUDA

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ai-coding-assistant.git](https://github.com/your-username/ai-coding-assistant.git)
cd ai-coding-assistant
