import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# =====================================================================
# 1. Hardware Configuration & Model Initialization
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting application on device: {device}")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer and model weights...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model weights onto the available hardware
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if device == "cpu":
    model.to("cpu")

print("Model loaded successfully!")

# =====================================================================
# 2. Prediction Function (Inference & Multi-Turn Memory)
# =====================================================================
def predict(message, history):
    """
    Formats multi-turn chat history into Qwen's role-based chat template
    and generates a clean, concise response.
    """
    # 1. Base system prompt defining assistant persona
    messages = [
        {
            "role": "system",
            "content": "You are a concise, helpful, and polite AI coding assistant. Provide clear, direct code examples and explanations."
        }
    ]

    # 2. Reconstruct past conversational context
    if history:
        for turn in history:
            # Handle standard Gradio 2-element list/tuple: [user_msg, bot_msg]
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                user_turn, bot_turn = turn
                if user_turn:
                    messages.append({"role": "user", "content": str(user_turn).strip()})
                if bot_turn:
                    messages.append({"role": "assistant", "content": str(bot_turn).strip()})
            # Handle dictionary-based turns if present
            elif isinstance(turn, dict):
                role = str(turn.get("role", "user"))
                content = str(turn.get("content", ""))
                if content.strip():
                    messages.append({"role": role, "content": content.strip()})

    # 3. Add the current incoming user question
    messages.append({"role": "user", "content": str(message).strip()})

    # 4. Apply Qwen's official chat template into raw text
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 5. Tokenize input string and send tensors to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 6. Configure text generation controls
    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.85,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 7. Generate response tokens
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            generation_config=gen_config
        )

    # 8. Extract and decode only newly generated tokens
    input_len = inputs.input_ids.shape[1]
    new_tokens = output_tokens[0][input_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =====================================================================
# 3. Gradio Web Interface
# =====================================================================
demo = gr.ChatInterface(
    fn=predict,
    title="AI Coding Assistant",
    description="An open-source conversational AI assistant powered by Qwen2.5-1.5B-Instruct and PyTorch.",
    examples=[
        "How do I reverse a linked list in Python?",
        "Explain what a Convolutional Neural Network does in simple terms.",
        "Write a quick binary search algorithm in C++."
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
