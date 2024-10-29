import json

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

ONE_TO_ONE_JSON_FILEPATH = "cfg/one_to_one.json"
one_to_one_map = json.load(open(ONE_TO_ONE_JSON_FILEPATH, "r"))


def custom_embed(text: str) -> list:
    lowered_text = text.lower()
    embedding_vector = []
    for i in range(len(one_to_one_map["data"])):
        keyword = one_to_one_map["data"][i].lower()
        if keyword in lowered_text:
            embedding_vector.append(1)
        else:
            embedding_vector.append(0)
    assert len(embedding_vector) == len(one_to_one_map["data"])
    return embedding_vector


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return_list = list()
        for document in input:
            return_list.append(custom_embed(document))
        return return_list


custom_embed_func = MyEmbeddingFunction()


def add_docs(coll):
    resp = coll.add(
        documents=[
            "This is a document about French restaurants.",
            "This is a document about bananas.",
        ],
        ids=["id1", "id2"],
    )
    return resp


def main():
    print("Hello from chromadb-search!")

    chroma_client = chromadb.Client()
    coll = chroma_client.create_collection(
        name="my_collection", embedding_function=custom_embed_func
    )
    add_docs(coll)
    results = coll.query(query_texts=["This talks about oranges."], n_results=2)
    print(results)
    return coll


if __name__ == "__main__":
    coll = main()
    # print(add_docs(coll))
