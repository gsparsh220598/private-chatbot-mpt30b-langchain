from langchain.cache import RedisCache, RedisSemanticCache
import langchain
from embeddings import *

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = "redis://localhost:6379"
emb_type = os.environ.get("EMB_TYPE")

# TODO: ADD FIX FOR THIS, THIS IS NOT WORKING
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379", embedding=get_embeddings(emb_type)
)
