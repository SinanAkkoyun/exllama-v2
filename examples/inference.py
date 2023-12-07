
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

model_directory =  "/home/ai/ml/llm/models/deepseek/coder-1.3B-base/exl2/3.0bpw"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
config.scale_pos_emb = 4

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)


# Generate some text

settings = ExLlamaV2Sampler.Settings()

settings.temperature = 1
settings.top_k = 1
settings.top_p = 0
settings.typical = 0
settings.token_repetition_penalty = 1.05
# settings.token_repetition_range = 5
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
"""
settings.temperature = 1
settings.top_k = 50
settings.top_p = 0.6
settings.token_repetition_penalty = 1
# 
"""

#prompt = """<｜begin▁of▁sentence｜><｜fim▁begin｜>#!/usr/bin/env python3
## get openai completion API in python
#<｜fim▁hole｜>
#<｜fim▁end｜>"""

prompt = """Once upon a time, there """

max_new_tokens = 500



"""
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.warmup()

time_begin = time.time()

input_ids = tokenizer.encode(prompt)
prompt_tokens = input_ids.shape[-1]

generator.set_stop_conditions([tokenizer.eos_token_id])
generator.begin_stream(input_ids, settings)

response_text = ""
gen_tokens = 0

while True:
    # Get response stream
    chunk, eos, tokens = generator.stream()
    
    gen_tokens += 1
    
    print(chunk, end="", flush=True)

    response_text += chunk

    
    if gen_tokens >= max_new_tokens:
        break
    
    if eos:
        break


time_end = time.time()
time_total = time_end - time_begin



print()
print(f"Response generated in {time_total:.2f} seconds, {gen_tokens} tokens, {gen_tokens / time_total:.2f} tokens/second")
"""

print("\n\n\n\n\n\n\nComparing to base")

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

generator.warmup()
time_begin = time.time()

output = generator.generate_simple(["Hi, ", "As usual,"], settings, max_new_tokens)

time_end = time.time()
time_total = time_end - time_begin

# newoutput is a sliced string of output at position (length of prompt, length of output)
# newoutput = output[len(prompt)-5:]

# get num tok of newoutput with encoding tokenzier
# num_tokens = tokenizer.encode(newoutput).shape[-1]


#print(newoutput)
print(output)
print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {(max_new_tokens * 70) / time_total:.2f} tokens/second")

