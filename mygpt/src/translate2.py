#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import json
from openai import OpenAI

TEXT_FILES = "../texts/"
OPENAI_KEY_FILE = "../keys/chat_gpt_anton.sec"

FILE_INPUT = '13 o.txt'
FILE_OUTPUT = FILE_INPUT[:-4] + ' rus.txt'

def is_float(s):
    try:
        _ = float(s)
        return True
    except:
        return False

with open(OPENAI_KEY_FILE, 'r') as f:
    openai_key = f.read().strip()

with open(TEXT_FILES + FILE_INPUT, 'r', encoding='utf-8') as f:
    input_text = f.read()

# prepare chatGPT input
input_text_list = input_text[1:].split("\n")

client = OpenAI(api_key=openai_key)
initial_message = "Suppose you are a wise rabbi. Translate the following from Hebrew to Russian. Don't repeat Hebrew phrases."

output_text_parts = []
completion_tokens = 0
prompt_tokens = 0
total_tokens = 0

terminator = "transcript"

for i, p in enumerate(input_text_list):
    if i < 5 or len(p) < 1 or p[0] == "{":
        output_text_parts.append(p)
        continue
    if terminator in p:
        output_text_parts.append(p)
        continue

    messages = [{"role": "system", "content": initial_message}]
    messages.append({"role": "user", "content": p})
        
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.01,
        messages=messages,
    )

    # completion.choices[0].message.content

    output_text_parts.append(completion.choices[0].message.content)
    
    completion_tokens += completion.usage.completion_tokens
    prompt_tokens += completion.usage.prompt_tokens
    total_tokens += completion.usage.total_tokens

ii = 1
indent = "\t"
with open(TEXT_FILES + FILE_OUTPUT, 'w', encoding='utf-8') as f:
    for p in output_text_parts:
        f.write(p + "\r\n")

print("completion_tokens:", completion_tokens)
print("prompt_tokens:", prompt_tokens)
print("completion_token price:", completion_tokens/1000000*5)
print("prompt_tokens price USD", prompt_tokens/1000000*15)
print("total price USD", completion_tokens/1000000*5 + prompt_tokens/1000000*15)
