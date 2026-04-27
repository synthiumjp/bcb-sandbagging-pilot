import json
from run_inference import load_model, format_system_prompt, format_user_message, run_single_trial, parse_response

items = []
with open('data/sampled_items.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        items.append(json.loads(line))

llm = load_model('llama3-8b')

for item in items:
    sys_prompt = format_system_prompt('A', item['domain'])
    user_msg = format_user_message(item)
    raw = run_single_trial(llm, sys_prompt, user_msg)
    parsed, valid = parse_response(raw)
    correct = parsed == item['answer'] if valid else None
    print(f"{item['domain']:12s} key={item['answer']} got={parsed} valid={valid} correct={correct} raw={repr(raw[:60])}")