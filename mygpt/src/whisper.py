#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import json
from openai import OpenAI

OPENAI_KEY_FILE = "/www/mygpt/keys/chat_gpt_anton.sec"
with open(OPENAI_KEY_FILE, 'r') as f:
    openai_key = f.read().strip()

input_audio = "/home/incoming/9d01470de62047d4bc120f57ab2c6496.mp3"
input_audio = "/home/incoming/me1.mp3"

audio_file = open(input_audio, "rb")

client = OpenAI(api_key=openai_key)

transcript1 = client.audio.transcriptions.create(
  model="whisper-1",
  language="ru",
  file=audio_file,
)


# prompt_file = "/home/incoming/b69ecede353b4003b277f1fe26031549.summary.txt"
# with open(prompt_file, "r") as f:
    # prompt = f.read()
# transcript2 = client.audio.transcriptions.create(
#   model="whisper-1",
#   file=audio_file,
#   prompt=prompt
# )
# tran2 = transcript2.text



# import requests
# import io
# url = 'https://api.openai.com/v1/audio/transcriptions'
# headers = {
#     'Authorization': f'Bearer {openai_key}'
# }


# audio_data = audio_file.read()
# buffer = io.BytesIO(audio_data)
# buffer.name = "audio.mp3"

# files = {
#     'file': ('audio.mp3', buffer, 'audio/mpeg')
# }

# data = {
#     'model': 'whisper-1',
#     'language': 'en'
# }

# response = requests.post(url, headers=headers, files=files, data=data)
# response.text


# text_input = "Our slogan is...   Learning is fun!..   Learning is health!..   Learning is life!.."

text_input = "This is a story about a new, AI-driven educational platform. [pause]"
text_input += "Our platform uses AI extensively to simplify knowledge sharing for social science experts, humanitarians, psychologists [pause]"
text_input += "Our platform includes AI advisers, remembering listeners circumstances and helping them for a long time [pause]"

text_input = "Sharing knowledge: lectures, recipes, etc."

# text_input = "We solve our customers problems - and entertain them. [pause]"
# text_input += "Imagine someone facing a pivotal life stage: Entering their later years, Navigating relationship challenges,"
# text_input += "Changing careers, Moving to a new country, Or dealing with various other life disruptions. [pause]"
# text_input += "Many people in these situations find it difficult to openly discuss their needs. However, they would benefit from a trusted, private confidant to provide support and guidance. [pause]"
# text_input += "Also, there are a lot of people who love to study something new, just for fun. [pause]"
# text_input += "And there are billions of such people. [pause]"
# text_input += ". [pause]"

# text_input = "Well, the platform simplifies the life for educators, OK. [pause]"
# text_input += "An obvious question: Why not to use AI only, without all these boring humans? [pause]"
# text_input += "Unfortunately, Internet-trained AI represents Internet knowledge. It often incorrect, biased or inconsistent, even hallucinating. [pause]"
# text_input += "In our platform all knowledge is shared from human to human - or human-written books, maybe. [pause]"

# text_input = "The last but not the least - what is our platform name? [pause]"
# text_input += "We think it can be Kvasir - the Norse god of wisdom and knowledge. [pause]"
# text_input += "Or it can be Anansi, a wise spider. [pause]"
# text_input += "Or even When Kvasir meets Anansi. [pause]"
# text_input += "The final choice will be done with the basic team complition. [pause]"

output_audio = '/media/Data/edtech/tmp/shimmer_40.mp3'


response = client.audio.speech.create(
    model="tts-1-hd",
    voice="shimmer",
    input=text_input,
    speed=1
)
response.stream_to_file(output_audio)
