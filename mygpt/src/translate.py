#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import json
from openai import OpenAI

TEXT_FILES = "../texts/"
OPENAI_KEY_FILE = "../keys/chat_gpt_anton.sec"

FILE_INPUT = 'rav Sherki 2024-9-5 num.txt'
FILE_OUTPUT = 'rav Sherki 2024-9-5 num rus.txt'

with open(OPENAI_KEY_FILE, 'r') as f:
    openai_key = f.read().strip()

with open(TEXT_FILES + FILE_INPUT, 'r', encoding='utf-8') as f:
    input_text = f.read()

# prepare chatGPT input
input_text_list = input_text[1:].split("\n")

input_text_parts = []
input_text_part = []
part_header = ("", "")
ii = 0
for l in input_text_list:
    ll = l.strip().split("\t")
    if len(ll) < 2:
        continue
    if "Speaker" in ll[1]:
        input_text_parts.append((part_header, input_text_part))
        part_header = (ll[0], ll[1])
        input_text_part = []
    else:
        input_text_part.append(ll)
input_text_parts.append((part_header, input_text_part))

client = OpenAI(api_key=openai_key)
initial_message = "Suppose you are a wise rabbi. Translate the following from Hebrew to Russian. Don't repeat Hebrew phrases."

output_text_parts = []
completion_tokens = 0
prompt_tokens = 0
total_tokens = 0
for p in input_text_parts:
    messages = [{"role": "system", "content": initial_message}]
    for l in p[1]:
        messages.append({"role": "user", "content": l[1]})
        
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.01,
        messages=messages,
    )

    # completion.choices[0].message.content

    output_text_parts.append((p[0],[q for q in completion.choices[0].message.content.split("\n") if len(q)>0]))
    completion_tokens += completion.usage.completion_tokens
    prompt_tokens += completion.usage.prompt_tokens
    total_tokens += completion.usage.total_tokens

ii = 1
indent = "\t"
with open(TEXT_FILES + FILE_OUTPUT, 'w', encoding='utf-8') as f:
    for p in output_text_parts:
        if p[0][0]:
            f.write(indent + "{:0}.".format(ii) + "\t" + p[0][1] + "(" + p[0][0] + ")"+ "\r\n")
            ii += 1
        for pp in p[1]:
            f.write(indent + "{:0}.".format(ii) + "\t" + pp + "\r\n")
            ii += 1

print("completion_tokens:", completion_tokens)
print("prompt_tokens:", prompt_tokens)
print("completion_token price:", completion_tokens/1000000*5)
print("prompt_tokens price USD", prompt_tokens/1000000*15)
print("total price USD", completion_tokens/1000000*5 + prompt_tokens/1000000*15)
