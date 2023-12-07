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
    ExLlamaV2Sampler
)

import torch
import time
import random

# Initialize model

model_directory =  "/home/ai/ml/llm/models/deepseek/coder-1.3B-base/exl2/3.0bpw"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
config.scale_pos_emb = 4

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

model.load()

tokenizer = ExLlamaV2Tokenizer(config)

# Create some sampling settings

settings_proto = ExLlamaV2Sampler.Settings()
settings_proto.temperature = 1
settings_proto.top_p = 0
settings_proto.top_k = 1
settings_proto.typical = 0
settings_proto.token_repetition_penalty = 1.05
# settings_proto.mirostat = True
# settings_proto.mirostat_tau = 5
# settings_proto.top_k = 1000

# Define some prompts to inference in parallel

strprompt = """### Book:
4
Transformations & Observables
In §2.1 we associated an operator with every observable quantity through
a sum over all states in which the system has a well-defined value of the
observable (eq. 2.5). We found that this operator enabled us to calculate
the expectation value of any function of the observable. Moreover, from the
operator we could recover the observable’s allowed values and the associ-
ated states because they are the operator’s eigenvalues and eigenkets. These
properties make an observable’s operator a useful repository of information
about the observable, a handy filing system. But they do not give the opera-
tor much physical meaning. Above all, they don’t answer the question ‘what
does an operator actually do when it operates?’ In this chapter we answer
this question. In the process of doing this, we will see why the canonical
commutation relations (2.54) have the form that they do, and introduce the
angular-momentum operators, which will play important roles in the rest of
the book.
4.1 Transforming kets
When one meets an unfamiliar object, one may study it by moving it around,
perhaps turning it over in one’s hands so as to learn about its shape. In §1.3
we claimed that all physical information about any system is encapsulated
in its ket |ψ〉, so we must learn how |ψ〉 changes as we move and turn the
system.
Even the simplest systems can have orientations in addition to posi-
tions. For example, an electron, a lithium nucleus or a water molecule all
have orientations because they are not spherically symmetric: an electron
is a magnetic dipole, a 7Li nucleus has an electric quadrupole, and a water
molecule is a V-shaped thing. The ket |ψ〉 that describes any of these objects
contains information about the object’s orientation in addition to its position
and momentum. In the next subsection we shall focus on the location of a
quantum system, but later we shall be concerned with its orientation as well,
and in preparation for that work we explicitly display a label μ of the sys-
tem’s orientation and any other relevant properties, such as internal energy.
For the moment μ is just an abstract symbol for orientation information; the
details will be fleshed out in §7.1.


### Question:
Why is the operator of an observable in quantum mechanics important for understanding the observable's properties?
### Response:
"""
prompts = [strprompt] * 70


max_parallel_seqs = (7 * 10) / 2

# Active sequences and corresponding caches and settings

input_ids = []
caches = []
settings = []

# Stats

total_gen_tokens = 0
total_prompt_tokens = 0
prompt_time = 0
token_time = 0

# Continue generating as long as there is work to do

while len(prompts) or len(input_ids):

    # If doing less than max_parallel_seqs, start some more. Prompt processing isn't batched in this example, but
    # would benefit much less from batching anyway

    while len(input_ids) < max_parallel_seqs and len(prompts):

        time_begin = time.time()

        prompt = prompts.pop()
        ids = tokenizer.encode(prompt)
        cache = ExLlamaV2Cache(model, max_seq_len = 573+500)  # (max_seq_len could be different for each cache)

        model.forward(ids[:, :-1], cache, preprocess_only = True)
        input_ids.append(ids)
        caches.append(cache)
        settings.append(settings_proto.clone())  # Need individual settings per prompt to support Mirostat

        total_prompt_tokens += ids.shape[-1] -1
        prompt_time += time.time() - time_begin

    # Create a batch tensor of the last token in each active sequence, forward through the model using the list of
    # active caches rather than a single, batched cache. Then sample for each token indidividually with some
    # arbitrary stop condition

    time_begin = time.time()

    inputs = torch.cat([x[:, -1:] for x in input_ids], dim = 0)
    logits = model.forward(inputs, caches, input_mask = None).float().cpu()

    eos = []
    r = random.random()
    for i in range(len(input_ids)):

        token, _, _ = ExLlamaV2Sampler.sample(logits[i:i+1, :, :], settings[i], input_ids[i], r, tokenizer)
        input_ids[i] = torch.cat([input_ids[i], token], dim = 1)
        total_gen_tokens += 1

        # if token.item() == tokenizer.newline_token_id or caches[i].current_seq_len == caches[i].max_seq_len:
        if caches[i].current_seq_len == caches[i].max_seq_len:
            eos.insert(0, i)

    token_time += time.time() - time_begin

    # Output and drop any sequences completed in this step

    for i in eos:

        output = tokenizer.decode(input_ids[i])[0]
        print("-----")
        print(output.strip())

        input_ids.pop(i)
        caches.pop(i)
        settings.pop(i)

# Stats

print("-----")
print(f"Prompts: {total_prompt_tokens} tokens, {total_prompt_tokens / prompt_time:.2f} tokens/second")
print(f"Tokens: {total_gen_tokens} tokens, {total_gen_tokens / token_time:.2f} tokens/second")

