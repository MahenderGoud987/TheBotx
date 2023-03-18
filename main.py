from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from flask import Flask, request, jsonify

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2/')

@app.route('/predict', methods=['POST'])

def generate_response():
    input_text = request.json['input_text']

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids=input_ids)
    return tokenizer.decode(output[0])

if __name__ == '__main__':
    app.run(debug=True)


