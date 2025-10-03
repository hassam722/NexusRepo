from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import os

model_id = "TheBloke/NexusRaven-V2-13B-GPTQ"
token = os.environ.get("HF_TOKEN")  # agar repo private ho

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)  # token specify karo agar private
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True, use_auth_token=token)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat(prompt):
    return pipe(prompt, max_new_tokens=200, temperature=0.7)[0]["generated_text"]

demo = gr.Interface(fn=chat, inputs="text", outputs="text")
demo.launch()
