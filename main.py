from sentence_transformers import SentenceTransformer
from redis import Redis
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.query import Query
import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, RangeQuery
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer
import os

# redis-py ---------------------------------------------------------------------------------------------------------------

# Pre-train Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

r = Redis(host="localhost", port=6379, db=0)

# text = "Understanding vector search is easy, but understanding all the mathematics behind a vector is not!"
# embedding = model.encode(text)
# print(embedding)

# r = Redis(host='localhost', port=6379, db=0)
# r.set('foo', 'bar')
# print(r.get('foo'))

# blob = embedding.astype(np.float32).tobytes()
# r.hset('doc:1', mapping = {'embedding': blob,
#                            'genre': 'technical',
#                            'content': text})

# vector = embedding.tolist()
# doc = {
#     'embedding': vector,
#     'genre': 'technical',
#     'content': text
# }
# r.json().set("doc:2", '$', doc)

# ------------------------------------------------------------------------------------

# Create the index
# index_def = IndexDefinition(prefix=["doc:"])
# schema = (
#     TextField("content", as_name="content"),
#     TagField("genre", as_name="genre"),
#     VectorField(
#         "embedding",
#         "FLAT",
#         {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"},
#     ),
# )
# r.ft("vec_idx").create_index(schema, definition=index_def)

# Import sample data
# r.hset(
#     "doc:1",
#     mapping={
#         "embedding": model.encode("That is a very happy person")
#         .astype(np.float32)
#         .tobytes(),
#         "genre": "persons",
#         "content": "That is a very happy person",
#     },
# )
# r.hset(
#     "doc:2",
#     mapping={
#         "embedding": model.encode("That is a happy dog").astype(np.float32).tobytes(),
#         "genre": "pets",
#         "content": "That is a happy dog",
#     },
# )
# r.hset(
#     "doc:3",
#     mapping={
#         "embedding": model.encode("Today is a sunny day").astype(np.float32).tobytes(),
#         "genre": "weather",
#         "content": "Today is a sunny day",
#     },
# )

# This is the test sentence
sentence = "That is a happy person"
# sentence = "This is a happy man"
# sentence = "The happy woman"
# sentence = "dog"
# sentence = "cat"
# sentence = "human"
# sentence = "child"

# ------------------------------------------------------------------------------------

q = (
    Query("*=>[KNN 2 @embedding $vec AS score]")
    .sort_by("score", asc=True)
    .return_field("score")
    .return_field("content")
    .dialect(2)
)
res = r.ft("vec_idx").search(
    q, query_params={"vec": model.encode(sentence).astype(np.float32).tobytes()}
)

# ------------------------------------------------------------------------------------

# radius = 0.2
# vec = model.encode(sentence).astype(np.float32).tobytes()
# q = (
#     Query("@embedding:[VECTOR_RANGE $radius $vec]=>{$YIELD_DISTANCE_AS: score}")
#     .return_fields("score", "content")
#     .dialect(2)
# )
# query_params = {"radius": radius, "vec": vec}
# res = r.ft("vec_idx").search(query=q, query_params=query_params)

# ------------------------------------------------------------------------------------

# q = (
#     Query("@genre:{pets}=>[KNN 2 @embedding $vec AS score]")
#     .return_field("score")
#     .return_field("content")
#     .dialect(2)
# )
# res = r.ft("vec_idx").search(
#     q, query_params={"vec": model.encode(sentence).astype(np.float32).tobytes()}
# )

print(res)


# redisvl ---------------------------------------------------------------------------------------------------------------
# index = SearchIndex.from_yaml("schema.yml")
# index.connect(os.environ.get('REDIS_URL', "redis://localhost:6379"))

# initialize the embedder
# hf = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

# create the index in Redis
# index.create(overwrite=True)

# load data into the index in Redis (list of dicts)
# data = [
#     {'content': 'That is a very happy person', 'genre': 'persons', 'embedding': hf.embed('That is a very happy person', as_buffer=True)},
#     {'content': 'That is a happy dog', 'genre': 'pets', 'embedding': hf.embed('That is a happy dog', as_buffer=True)},
#     {'content': 'Today is a sunny day', 'genre': 'weather', 'embedding': hf.embed('Today is a sunny day', as_buffer=True)}
# ]

# index.load(data)

# perform the VSS query
# query = VectorQuery(
#     vector=hf.embed('That is a happy person'),
#     vector_field_name="embedding",
#     return_fields=["content"],
#     num_results=3,
#     return_score=True,
# )

# query = RangeQuery(
#     vector=hf.embed('That is a happy person'),
#     vector_field_name="embedding",
#     return_fields=["score", "content"],
#     dtype="float32",
#     distance_threshold=0.1,
#     return_score=True,
#     dialect=2,
# )

# results = index.query(query)
# print(results)
