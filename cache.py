from langchain.cache import RedisCache, RedisSemanticCache, SQLiteCache, InMemoryCache
import langchain
from embeddings import *

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = "redis://localhost:6360"
emb_type = os.environ.get("EMB_TYPE")

# TODO: ADD FIX FOR THIS, THIS IS NOT WORKING
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379", embedding=get_embeddings(emb_type)
)

# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# langchain.llm_cache = InMemoryCache()
