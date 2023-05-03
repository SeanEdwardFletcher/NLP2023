import requests
import csv
import json

# directory to save the top-2000(ish) results per query using elastic search API
the_directory = 'put your directory here'

user = 'your user name here'
password = 'your password here'

def download_elastic(target_dir, num_of_results):
    # Reading the topic file
    with open("SP12023topics.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        # skip header
        next(reader)

        pre_qid = ""
        counter = 1
        for line in reader:
            query_to_es = line[-1]
            url = query_to_es + "&size=" + str(num_of_results)
            q_id = line[0]
            if q_id == pre_qid:
                counter += 1
            else:
                counter = 1
            pre_qid = q_id
            result = requests.get(url, auth=(user, password)).content.decode("utf-8")
            obj = json.loads(result)

            # remove quote
            hits_count = len(obj['hits']['hits'])
            if hits_count < num_of_results:
                if query_to_es.endswith("\""):
                    original_query = query_to_es.split("q=")[1][1:-1]
                    remaining = num_of_results - hits_count
                    remade_query = query_to_es.split("q=")[0] + "q=" + original_query + "&size=" + str(remaining)
                    result = requests.get(remade_query, auth=(user, password)).content.decode("utf-8")
                    obj2 = json.loads(result)
                    for item in obj2['hits']['hits']:
                        if item not in obj['hits']['hits']:
                            obj['hits']['hits'].append(item)

            # # replace with topic_text
            # hits_count = len(obj['hits']['hits'])
            # if hits_count < 2000:
            #     original_query = query_to_es.split("q=")[1][1:-1]
            #     remaining = 2000 - hits_count
            #     remade_query = query_to_es.split("q=")[0] + "q=" + line[1] + "&size=" + str(remaining)
            #     result = requests.get(remade_query, auth=(user, password)).content.decode("utf-8")
            #     obj2 = json.loads(result)
            #     for item in obj2['hits']['hits']:
            #         if item not in obj['hits']['hits']:
            #             obj['hits']['hits'].append(item)

            # result = ast.literal_eval(result)
            with open(target_dir + q_id + "_" + str(counter) + ".json", "w") as file:
                json.dump(obj, file)

