from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def get_embedding(model, text):
    # This method takes in the model and the input text
    # and returns a 768-dimensional embedding of the text
    embedding = model.encode([text])
    return embedding


def get_s_bert_top_100_matches(dict_of_json_files, dict_of_queries, model):
    top_100_dict = {}
    for query_id, triple00 in tqdm(dict_of_queries.items()):
        q_t = triple00[0] + triple00[1]  # q_t means query text: title + body
        q_t_embedding = [model.encode(q_t)]

        for query_id00, dict00 in tqdm(dict_of_json_files.items()):
            reformatted_key = query_id00[:3]
            top_100_list = []

            for doc_id, tuple00 in dict00.items():
                text = tuple00[0] + tuple00[1]
                doc_embedding = [model.encode(text)]
                co_sim = cosine_similarity(q_t_embedding, doc_embedding)

                if len(top_100_list) < 100:
                    top_100_list.append((doc_id, co_sim))
                    top_100_list = sorted(top_100_list, key=lambda x: x[1], reverse=True)
                else:
                    if co_sim > top_100_list[-1]:
                        top_100_list.append((doc_id, co_sim))
                        top_100_list = sorted(top_100_list, key=lambda x: x[1], reverse=True)
                        top_100_list = top_100_list[:100]

    return top_100_dict
