import chromadb
from chromadb.utils import embedding_functions


client = chromadb.CloudClient(...)
col = client.get_collection("tn_class6_text")

res = col.query(
    query_texts=["beach"],
    n_results=5,
    include=["documents", "metadatas"]
)

print(res)
