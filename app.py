import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./vector_store")

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="demo",
    embedding_function = embed_fn
)

collection.add(
    ids = ["1","2","3","4"],
    documents = [
        "Apache Airflow orchestration data pipelines.",
        "Photosynthesis converts sunlight into chemical energy",
        "Python is popular for machine learning",
        "The Big Ben is in London"
    ],
    metadatas = [
        {"topic":"data"},
        {"topic":"science"},
        {"topic":"tech"},
        {"topic":"travel"}
    ]
)

# Semantic Search
results = collection.query(
    query_texts = ["How do I schedule ETL workflows?"],
    n_results = 2
)

for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"{dist:.3f} {doc}")

# Filtering by metadata
print("*******************FILTERING*******************")
filtered = collection.query(
    query_texts=["learning from data"],
    n_results = 1,
    where = {"topic":"tech"}
)
print(filtered["documents"][0]) 