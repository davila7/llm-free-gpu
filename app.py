from pyngrok import ngrok

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.5, device=None)
torch.backends.cuda.reserved_memory.max_split_size_mb = 1024

from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

import base64
from io import BytesIO

# Load text model
text_model_id = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(text_model_id, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(text_model_id, trust_remote_code=True)

# Load text model
# image_model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(image_model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

import streamlit as st
def main():
  st.title("LLM phi-2 with Streamlit")

  prompt = st.text_input("Enter a prompt", "")

  if st.button("Generate"):
      # # generate image
      # image = pipe(prompt).images[0]
      # st.image(image, caption="Generated Image")

      # generate text
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
      generated_output = model.generate(input_ids, do_sample=True, temperature=1.0, max_length=2500, num_return_sequences=1)
      generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

      st.write("Generated Text:")
      st.write(generated_text)

if __name__ == '__main__':
    public_url = ngrok.connect(port='8501')
    print('Public URL:', public_url)
    main()