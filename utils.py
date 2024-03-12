import json

def readjson(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line)) # Load the string in this line of the file
    print(data[0])
    return data

#readjson("data/A2_val.jsonl")