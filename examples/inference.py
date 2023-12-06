
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache

model_directory =  "/home/ai/ml/llm/models/llama/code-7B/gptq"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
# config.scale_pos_emb = 4

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.05
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

prompt = """#!/usr/bin/env python3
# get openai completion
"""


max_new_tokens = 150

generator.warmup()
time_begin = time.time()

output = generator.generate_simple(prompt, settings, max_new_tokens)

time_end = time.time()
time_total = time_end - time_begin

print(output)
print()
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")



print("Comparing to streaming")

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
# generator.set_stop_conditions(prompt_format.stop_conditions(tokenizer))
generator.warmup()

time_begin = time.time()

input_ids = tokenizer.encode(prompt)
prompt_tokens = input_ids.shape[-1]

generator.set_stop_conditions([])
generator.begin_stream(input_ids, settings)

response_text = ""
gen_tokens = 0

while True:

    # Get response stream

    chunk, eos, tokens = generator.stream()
    response_text += chunk
    # responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim = -1)
    gen_tokens += 1

    # check if gentokens >= maxlength
    
    if gen_tokens >= max_new_tokens:
        break

    # Check to see if we've reached the end of the text

    if eos:
        break


time_end = time.time()
time_total = time_end - time_begin

print(output)
print()
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
