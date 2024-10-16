#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import json
from openai import OpenAI

TEXT_FILES = "../texts/"
OPENAI_KEY_FILE = "../keys/chat_gpt_anton.sec"

FILE_INPUT = 'leNevuchey-num-simpl.txt'
FILE_OUTPUT = 'leNevuchey-num-simpl-rus.txt'

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

input_text_parts = []
for l in input_text_list:
    ll = l.strip().split("\t")
    input_text_parts.append(ll)

client = OpenAI(api_key=openai_key)
initial_message = "Suppose you are a wise rabbi. Translate the following from Hebrew to Russian. Don't repeat Hebrew phrases."

output_text_parts = []
completion_tokens = 0
prompt_tokens = 0
total_tokens = 0
for p in input_text_parts:
    messages = [{"role": "system", "content": initial_message}]
    if len(p) < 1:
        output_text_parts.append(())
        continue
    elif len(p) == 1:
        if len(p[0]) == 0 or is_float(p[0]):
            output_text_parts.append((p[0], ))
            continue
        
        messages.append({"role": "user", "content": p[0]})
    else:
        if len(p[1]) == 0:
            output_text_parts.append((p[0], "" ))
            continue
        messages.append({"role": "user", "content": p[1]})
        
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.01,
        messages=messages,
    )

    # completion.choices[0].message.content

    if len(p) == 1:
        output_text_parts.append((completion.choices[0].message.content, ))
    else:
        output_text_parts.append((p[0], completion.choices[0].message.content))
    
    completion_tokens += completion.usage.completion_tokens
    prompt_tokens += completion.usage.prompt_tokens
    total_tokens += completion.usage.total_tokens

ii = 1
indent = "\t"
with open(TEXT_FILES + FILE_OUTPUT, 'w', encoding='utf-8') as f:
    for p in output_text_parts:
        if len(p) < 1:
            f.write("\r\n")
        elif len(p) == 1:
            f.write(p[0] + "\r\n")
        else:
            f.write(p[0] + "\t" + p[1] + "\r\n")

print("completion_tokens:", completion_tokens)
print("prompt_tokens:", prompt_tokens)
print("completion_token price:", completion_tokens/1000000*5)
print("prompt_tokens price USD", prompt_tokens/1000000*15)
print("total price USD", completion_tokens/1000000*5 + prompt_tokens/1000000*15)
