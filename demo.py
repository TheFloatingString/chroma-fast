import json
import time

import chromadb
import tqdm
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

ONE_TO_ONE_JSON_FILEPATH = "cfg/one_to_one.json"
one_to_one_map = json.load(open(ONE_TO_ONE_JSON_FILEPATH, "r"))

RESTAURANTS_FILEPATH = "data/sample_montreal_restaurants.json"
restaurant_data_raw = json.load(open(RESTAURANTS_FILEPATH, "rb"))

CONFIG_JSON_FILEPATH = "config.json"
config_json_data = json.load(open(CONFIG_JSON_FILEPATH, "r"))

VERBOSE = False


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
        return_list = list()
        for document in input:
            return_list.append(custom_embed(document))
        return return_list


custom_embed_func = MyEmbeddingFunction()


def add_docs(coll):
    if VERBOSE:
        print(restaurant_data_raw.keys())

    reviews_to_add = list()
    review_ids_to_add = list()

    counter = 0
    for i in range(len(restaurant_data_raw["places"])):
        RESTAURANT_ID = restaurant_data_raw["places"][i]["id"]
        for j in range(len(restaurant_data_raw["places"][i]["reviews"])):
            review = restaurant_data_raw["places"][i]["reviews"][j]["text"]["text"]
            review_id = f"{RESTAURANT_ID}-{j}"

            reviews_to_add.append(review)
            review_ids_to_add.append(review_id)
            counter += 1

    if VERBOSE:
        print(f"counter: {counter}")

    resp = coll.add(
        documents=reviews_to_add,
        ids=review_ids_to_add,
    )
    return resp


def main():
    if VERBOSE:
        print("Hello from chromadb-search!")

    chroma_client = chromadb.Client()
    if config_json_data["use_LLM_embeds"]:
        coll = chroma_client.create_collection(name="my_collection")
    else:
        coll = chroma_client.create_collection(
            name="my_collection", embedding_function=custom_embed_func
        )
    add_docs(coll)

    query = input("Describe the restaurant that you are looking for: ")

    start = time.time()
    results = coll.query(query_texts=[query], n_results=3)

    # print(results)

    result_ids = results["ids"][0]
    restaurant_ids = list(set([tmp.split("-")[0] for tmp in result_ids]))
    if VERBOSE:
        print(result_ids)
    print("Search results:")
    print("---------------")
    for initial_id in restaurant_ids:
        # restaurant_id = initial_id.split("-")[0]
        restaurant_id = initial_id
        for i in range(len(restaurant_data_raw["places"])):
            if restaurant_data_raw["places"][i]["id"] == restaurant_id:
                print(
                    f'{restaurant_data_raw["places"][i]["displayName"]["text"]} ({restaurant_data_raw["places"][i]["shortFormattedAddress"]})\t\t{restaurant_data_raw["places"][i]["websiteUri"]}'
                )
    end = time.time()
    print("---------------")
    print(f"Elapsed: {round(end-start, 8)} seconds")

    return coll


if __name__ == "__main__":
    coll = main()
