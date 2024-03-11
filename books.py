import redis
import io
import os
import json
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import TextField, TagField, NumericField, VectorField
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


# Get the recommendation from a book 
def get_recommendation(key):
    embedding = r.json().get(key)
    embedding_as_blob = np.array(embedding['embedding'], dtype=np.float32).tobytes()
    q = Query("*=>[KNN 5 @embedding $vec AS score]").return_fields("score", "$.author", "$.title", "$.description").sort_by("score", asc=True).dialect(2).paging(1, 5)
    res = r.ft("books_idx").search(q, query_params={"vec": embedding_as_blob})
    return res

def get_recommendation_by_range(key):
    embedding = r.json().get(key)
    embedding_as_blob = np.array(embedding['embedding'], dtype=np.float32).tobytes()
    q = Query("@embedding:[VECTOR_RANGE $radius $vec]=>{$YIELD_DISTANCE_AS: score}") \
        .return_fields("title") \
        .sort_by("score", asc=True) \
        .paging(1, 5) \
        .dialect(2)

    # Find all vectors within a radius from the query vector
    query_params = {
        "radius": 3,
        "vec": embedding_as_blob
    }

    res = r.ft("books_idx").search(q, query_params)
    return res

if __name__ == "__main__":
    
    REDIS_URL=os.getenv('REDIS_URL', "redis://localhost:6379")

    # Get a Redis connection
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    print(r.ping())

    # Define the model we want to use 
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    """
    # Reading the books and storing as JSON documents together with the corresponding vector embedding
    # The embedding is calculated using the description.
    for filename in os.listdir("./data/books/"):
        f = os.path.join("./data/books/", filename)

        if os.path.isfile(f):
            book_file = io.open(f, encoding="utf-8")
            book_json = json.load(book_file)
            book_file.close()
            print(f"{book_json['title']} done")
            book_json['embedding'] = model.encode(book_json['description']).astype(np.float32).tolist()
            r.json().set(f"book:{book_json['id']}", "$", book_json)
            #r.json().set(f"book:{book_json['id']}", "$.embedding", model.encode(book_json['description']).astype(np.float32).tolist())

    # Here we create an index if it does not exists
    indexes = r.execute_command("FT._LIST")
    if "books_idx" not in indexes: #.encode("utf-8")
        index_def = IndexDefinition(prefix=["book:"], index_type=IndexType.JSON)
        schema = (TagField("$.title", as_name="title"),
                TagField("$.status", as_name="status"),
                TagField("$.author", as_name="author"),
                NumericField("$.year_published", as_name="year_published"),
                VectorField("$.embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE", "INITIAL_CAP": 1500}, as_name="embedding"))
        r.ft('books_idx').create_index(schema, definition=index_def)
    """
    
    # print(get_recommendation('book:26415'))
    # print(get_recommendation('book:9'))

    print(get_recommendation_by_range('book:26415'))
    print(get_recommendation_by_range('book:9'))