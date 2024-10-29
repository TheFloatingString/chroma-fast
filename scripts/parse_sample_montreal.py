import json


def main():
    JSON_FILEPATH = "data/sample_montreal_restaurants.json"
    data = json.load(open(JSON_FILEPATH, "rb"))
    print(data.keys())
    print(len(data["places"]))
    print(data["places"][0].keys())
    print(data["places"][0]["reviews"][0].keys())
    print(data["places"][0]["reviews"][0]["text"]["text"])


if __name__ == "__main__":
    main()
