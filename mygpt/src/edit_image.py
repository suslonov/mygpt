#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
from flask import Flask, render_template, request, flash
from hashlib import md5
import json
from openai import OpenAI

OPENAI_KEY_FILE = "/www/mygpt/keys/chat_gpt_anton.sec"
with open(OPENAI_KEY_FILE, 'r') as f:
    openai_key = f.read().strip()

client = OpenAI(api_key=openai_key)

input_image = "/home/incoming/Pictures/anton.png"

res = client.images.edit(
  image = open(input_image, "rb"),
  prompt="draw smiling muppet similar to this picture, low details",
  n=2,
  size="512x512"
)

