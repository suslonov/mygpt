#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import json
from openai import OpenAI

TEXT_FILES = "../texts/"
OPENAI_KEY_FILE = "../keys/chat_gpt_anton.sec"

FILE_INPUT = 'Universal Religious Zionism HE.txt'
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

input_text_part = ("", [])
input_text_parts = []
completion_tokens = 0
prompt_tokens = 0
total_tokens = 0

end_chapter = False
new_line = False

for i, p in enumerate(input_text_list):
    if len(p) < 1:
        if new_line and i > 3:
            end_chapter = True
        else:
            new_line = True
    else:
        new_line = False
        
    if end_chapter:
        input_text_parts.append(input_text_part)
        input_text_part = ("\r\n", [])
        end_chapter = False
    if new_line:
        continue

    elif p[:3] == "***":
        input_text_parts.append(input_text_part)
        input_text_part = ("***\r\n", [])
    elif p[0] == "(" and p[2] == ")" :
        input_text_parts.append(input_text_part)
        input_text_part = (p[:3], [p[3:]])
    elif p[0].isnumeric():
        if p[1:3] == ". ":
            input_text_parts.append(input_text_part)
            input_text_part = (p[:3], [p[3:]])
        elif p[1:3] == ".\t":
            input_text_parts.append(input_text_part)
            input_text_part = (p[:3], [p[3:]])
        elif p[2:4] == ". ":
            input_text_parts.append(input_text_part)
            input_text_part = (p[:4], [p[4:]])
        elif p[1] == " ":
            input_text_parts.append(input_text_part)
            input_text_part = (p[:2], [p[2:]])
        elif p[2].isnumeric():
            if p[3] == ".":
                input_text_parts.append(input_text_part)
                input_text_part = (p[:4], [p[4:]])
            elif p[3] == " ":
                input_text_parts.append(input_text_part)
                input_text_part = (p[:3], [p[3:]])
    elif p[2].isnumeric() and p[1] == "-":
        if p[3] == ".":
            input_text_parts.append(input_text_part)
            input_text_part = (p[:4], [p[4:]])
        elif p[3] == " ":
            input_text_parts.append(input_text_part)
            input_text_part = (p[:3], [p[3:]])

    elif p[:6] == "------":
        input_text_parts.append(input_text_part)
        input_text_parts.append((p, []))
        break
    elif len(p) > 1:
        input_text_part[1].append(p)

output_text_parts = []
for p in input_text_parts:
    if len(p[1]) == 0:
        continue
    messages = [{"role": "system", "content": initial_message}]
    for pp in p[1]:
        if len(pp) > 0:
            messages.append({"role": "user", "content": pp})
        
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.01,
        messages=messages,
    )

    # completion.choices[0].message.content

    output_text_parts.append((p[0], [o for o in completion.choices[0].message.content.split("\n") if len(o) != 0]))
    
    completion_tokens += completion.usage.completion_tokens
    prompt_tokens += completion.usage.prompt_tokens
    total_tokens += completion.usage.total_tokens

with open(TEXT_FILES + FILE_OUTPUT, 'w', encoding='utf-8') as f:
    for p in output_text_parts:
        if len(p[0]) != 0:
            if p[0][-1] == "\n":
                f.write(p[0])
            else:
                f.write(p[0] + " ")
        for pp in p[1]:
            f.write(pp + "\r\n")

print("completion_tokens:", completion_tokens)
print("prompt_tokens:", prompt_tokens)
print("completion_token price:", completion_tokens/1000000*5)
print("prompt_tokens price USD", prompt_tokens/1000000*15)
print("total price USD", completion_tokens/1000000*5 + prompt_tokens/1000000*15)
