from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.set_default_device("cuda")

import base64
from io import BytesIO

# Load model
model_id = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate():
  #get the prompt input
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  #generate text
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  generated_output = model.generate(input_ids, do_sample=True, temperature=0.0, max_length=500, num_return_sequences=1)
  generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

  print("Sending image and text ...")
 
  return render_template('index.html', generated_text=generated_text, prompt=prompt)

if __name__ == '__main__':
    app.run()