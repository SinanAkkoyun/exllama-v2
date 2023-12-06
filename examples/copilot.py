# This is an example of a http steaming server support Github Copilot VSCode extension
# 1. `uvicorn copilot:app --reload --host 0.0.0.0 --port 9999`
# 2. Configure VSCode copilot extension (in VSCode's settings.json):
# ```json
# "github.copilot.advanced": {
#     "debug.overrideEngine": "engine", # can be any string.
#     "debug.testOverrideProxyUrl": "http://localhost:9999",
#     "debug.overrideProxyUrl": "http://localhost:9999"
# }
# ```

import sys, os
import time
from uuid import uuid4

# Global variable to track the current task
current_task_id = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from os import times
import logging
import json

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler, ExLlamaV2BaseGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from typing import List, Optional
from pydantic import BaseModel

log = logging.getLogger("uvicorn")
log.setLevel("DEBUG")
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """_summary_
    Starts up the server, setting log level, downloading the default model if necessary.

    Edited from https://github.com/chenhunghan/ialacol/blob/main/main.py
    """
    log.info("Starting up...")
    log.debug("Creating generator instance...")

    model_directory = "/home/ai/ml/llm/models/deepseek/coder-6.7B-base/gptq"
    draft_model_directory = "/home/ai/ml/llm/models/deepseek/coder-1.3B-base/exl2/3.0bpw"

    

    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()
    config.scale_pos_emb = 4        
    # config.max_seq_len = 8192
    tokenizer = ExLlamaV2Tokenizer(config)

    
    model = ExLlamaV2(config)
    log.debug("Loading model...")
    model.load([23, 0])
    cache = ExLlamaV2Cache(model)

    #draft_config = ExLlamaV2Config()
    #draft_config.model_dir = draft_model_directory
    #draft_config.prepare()  
    # draft_config.max_seq_len = 8192
    #draft_config.scale_pos_emb = 4

    #draft_model = ExLlamaV2(config)
    #log.debug("Loading draft model...")
    #draft_model.load([0, 23])
    #draft_cache = ExLlamaV2Cache(draft_model)

    log.debug("Creating generator instance...")
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)


    # Ensure CUDA is initialized
    log.debug("Warming up generator instance...")
    generator.warmup()
    app.state.generator = generator
    app.state.tokenizer = tokenizer
    log.debug("Generator instance created.")


class CompletionRequestBody(BaseModel):
    """_summary_
    from from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    prompt: str = ""
    suffix: str = ""
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.85
    top_p: Optional[float] = 0.8
    stop: Optional[List[str] | str] = ["\ndef ", "\nclass ", "\nif ", "\n\n#"]
    stream: bool = True
    model: str = ""
    top_k: Optional[int] = 50

    repetition_penalty: Optional[float] = 1

    class Config:
        arbitrary_types_allowed = True

def remove_leading_comments(code):
    lines = code.split('\n')
    cleaned_lines = []
    first_comment_found = False

    for line in lines:
        # Check if the line is a comment
        if line.startswith("# "):
            if not first_comment_found:
                # Keep the first comment
                cleaned_lines.append(line)
                first_comment_found = True
            # Skip the rest of the comments
            continue
        # Add non-comment lines
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


@app.post("/v1/engines/{engine}/completions")
async def engine_completions(
    # Can't use body as FastAPI require corrent context-type header
    # But copilot client maybe not send such header
    request: Request,
    # copilot client ONLY request param
    engine: str,
):
    """_summary_
        From https://github.com/chenhunghan/ialacol/blob/main/main.py

        Similar to https://platform.openai.com/docs/api-reference/completions
        but with engine param and with /v1/engines
    Args:
        body (CompletionRequestBody): parsed request body
    Returns:
        StreamingResponse: streaming response
    """
    global current_task_id      # Implemented stopping old completion requests

    # Update the task id for the new request
    new_task_id = str(uuid4())
    current_task_id = new_task_id


    req_json = await request.json()
    # log.debug("Body:%s", str(req_json))

    # print(json.dumps(req_json, indent=4))

    body = CompletionRequestBody(**req_json, model=engine)

    prompt = body.prompt
    suffix = body.suffix

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.95
    settings.top_k = 50
    settings.top_p = 0.7
    settings.token_repetition_penalty = 1.05                                           # DeepSeek models don't like repp

    tokenizer = app.state.tokenizer
    # settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    max_new_tokens = 200 # body.max_tokens if body.max_tokens else 1024

    generator = request.app.state.generator
    
    
    

    # Only for stream
    # generator.set_stop_conditions([tokenizer.eos_token_id])

    # DeepSeek insert prompt format
    #if suffix is not None:
    model_input_str = "<｜fim▁begin｜>" + remove_leading_comments(prompt) + "<｜fim▁hole｜>\n" + suffix + "<｜fim▁end｜>"
    input_ids = tokenizer.encode(model_input_str)

    print(model_input_str)
    # else:
    #    input_ids = tokenizer.encode(prompt)

    log.debug("Streaming response from %s", engine)


    def stream():
        print("\nNEW STREAM >\n")
        generator.current_seq_len = 0
        generator.begin_stream(input_ids, settings)
        generated_tokens = 0

        print("stream began")


        start_time = time.time()

        while True:
            if current_task_id != new_task_id:
                print("\n< NEW TASK\n")
                # A new request has come in, stop this generation
                break

            chunk, eos, _ = generator.stream()
            # log.debug("Streaming chunk %s", chunk)
            # print(chunk, end="")

            created = times()
            generated_tokens += 1
            if eos or generated_tokens == max_new_tokens:
                print(generated_tokens)
                print(time.time() - start_time)

                print(generated_tokens / (time.time() - start_time))

                stop_data = json.dumps(
                    {
                        "id": "id",
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": engine,
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                yield f"data: {stop_data}" + "\n\n"
                break
            data = json.dumps(
                {
                    "id": "id",
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": engine,
                    "choices": [
                        {
                            "text": chunk,
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
            )
            yield f"data: {data}" + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )
