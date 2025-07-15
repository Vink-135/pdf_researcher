from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")

def generate_answer_tf(question, context):
    prompt = f"question: {question} context: {context}"
    input_ids = tokenizer(prompt, return_tensors="tf", truncation=True, max_length=512).input_ids
    output = t5_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)