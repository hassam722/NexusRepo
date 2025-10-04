from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import os

model_id = "Nexusflow/NexusRaven-V2-13B"

tokenizer = AutoTokenizer.from_pretrained(model_id)  # token specify karo agar private
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat(prompt):
    return pipe(prompt, max_new_tokens=200, temperature=0.7)

demo = gr.Interface(fn=chat, inputs="text", outputs="text")
demo.launch(share= True)
