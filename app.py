# pip -q install git+https://github.com/huggingface/transformers # need to install from github
# pip install -q datasets loralib sentencepiece
# pip -q install xformers einops
# pip -q install hf_transfer
# pip install huggingface
# pip install accelerate
# pip install -U bitsandbytes


from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from huggingface_hub import login

app = Flask(__name__)
os.environ ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

login()

torch.set_default_device('cuda')
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it",
                                             quantization_config=quantization_config,
                                             low_cpu_mem_usage=True,
                                             torch_dtype="auto",
                                             device_map="auto"
                                             )

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
def generate(age, gender):
    input_text=f"write a poem for child whose age is {age} and gender is {gender}"
    system_prompt = "You are a poet you write poems for childs"
    messages = [
        {"role": "user", "content": system_prompt + '\n\n' +input_text},
    ]

    prompt = tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device),
                             max_new_tokens=512,
                             do_sample=True,
                             temperature=0.1,
                             top_k=50,
                             )
    text = tokenizer.decode(outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text = text.replace('user\n'+system_prompt+ '\n\n' +input_text+ '\nmodel', '', 1)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        result = generate(age, gender)
    
    return render_template('index.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)
