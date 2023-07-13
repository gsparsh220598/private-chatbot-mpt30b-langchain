from langchain.cache import RedisCache, RedisSemanticCache
import langchain
from embeddings import *

REDIS_URL = "redis://localhost:6379"

langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379", embedding=get_embeddings(emb_type="openai")
)
