from dotenv import dotenv_values

from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

from constants import (EMBEDDING_MODEL,
                       LLM_MODEL,
                       RERANK_MODEL)

config = dotenv_values(".env")

# llama-3.2-nv-embedqa-1b-v1
embedding_model = NVIDIAEmbedding(base_url=config["NVIDIA_URL"], 
                                  api_key=config["EMBED_MODEL_API_KEY"],
                                  model=EMBEDDING_MODEL,
                                  truncate="NONE")

# llama-3.2-1b-instruct
llm = NVIDIA(base_url=config["NVIDIA_URL"],
             api_key=config["LLM_API_KEY"],
             model=LLM_MODEL)

# llama-3.2-nv-embedqa-1b-v1
rerank_model = NVIDIARerank(api_key=config["RERANK_MODEL_API_KEY"],
                            model=RERANK_MODEL,
                            top_n=5)
