
from langchain_community.utilities import SerpAPIWrapper

search = SerpAPIWrapper(serpapi_api_key="")  # Obtén la clave en https://serpapi.com/

def search_web(query, num_results=3):
    results = search.run(query)
    return results[:num_results]


