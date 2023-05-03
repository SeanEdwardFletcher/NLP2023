from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from Preprocessing_tools import *
from get_article_content import *
from download_elastic import download_elastic
from torch.utils.data import DataLoader
from sentence_bert_tools import *



directory_of_large_json_files_from_elastic_search = 'first directory path here'
directory_of_json_files_from_the_qrel_file = 'second directory path here'
directory_of_top_100_query_results = 'third directory path here'
simple_text_training_data_path = 'path to training data file here'


# ***********************************************************************************************
# this call uses the API provided by Simple Text to call each query provided by simple text
# if the initial query doesn't return the desired number of results, it's called again but without quotation marks
download_elastic(directory_of_large_json_files_from_elastic_search, 500)
download_elastic(directory_of_top_100_query_results, 100)


# ***********************************************************************************************
# splitting the qrel file into separate training and validation qrel files
the_qrel_file_from_SimpleText = simple_text_training_data_path
training_qrel_file_name = "Seans_qrel_file_training"
validation_qrel_file_name = "Seans_qrel_file_validation"
test_qrel_file_name = "Seans_qrel_file_test"
split_the_qrel_file(
    the_qrel_file_from_SimpleText,
    training_qrel_file_name,
    validation_qrel_file_name,
    test_qrel_file_name)


# ***********************************************************************************************
# downloading json files that correspond to all the articles in the qrel file
qrel_file_path = simple_text_training_data_path
qrel_file = open(qrel_file_path, "r")
for line in qrel_file:
    # Split the line into its components (query ID, a zero, document ID, and relevance score)
    components = line.split()
    document_id = components[2]  # document ID
    # make an elastic search call, save the result as a json, put that json in a directory
    make_elastic_call_for_single_article(document_id, directory_of_json_files_from_the_qrel_file)


# ***********************************************************************************************
# getting the article title, body, and keywords from The Guardian
dict_of_query_IDs_and_urls = get_the_gardians_queryIDs_and_urls("SP12023topics.csv")  # {"query_id": "article_url"}
dict_of_TheGuardian_article_content = {}  # {"G01": ("article_title", "article_body", ["the", "article's", "keywords])}
for id_00, url_00 in dict_of_query_IDs_and_urls.items():
    first_3_char_00 = id_00[:3]
    if first_3_char_00 in dict_of_TheGuardian_article_content.keys():
        continue
    else:
        dict_of_TheGuardian_article_content[first_3_char_00] = get_theguardians_content(url_00=url_00)


# ***********************************************************************************************
# this block gathers all the query_articles' keywords
keyword_dict_gold_standard_tg = {}  # tg stands for The Guardian.
                                    # {'G01.1': ["the word/phrase used in the elastic search query",
                                    #            "all", "the keywords", "from tg's website's", "html code"]
keyword_dict_yake_tg = {}  # tg stands for The Guardian.
                           # {'G01.1': ['a list', 'of', 'keywords produced', 'by', 'YAKE']}
keyword_dict_combined = {} # {'G01.1': ['all the', 'keywords']}
keyword_dict_yake_plus_queryterm = {}

# filling gold_standard_tg_keyword_dict:
dict_of_query_terms = get_the_query_search_term_from_the_simpletext_csv_file("SP12023topics.csv")  # {'query_id': 'the search query'}
for key, item in dict_of_query_terms.items():
    first_3_char_01 = key[:3]
    tg_article_kws = dict_of_TheGuardian_article_content[first_3_char_01][2]
    query_keyword = item
    query_keyword_as_list = query_keyword.split()
    if len(query_keyword_as_list) > 2:  # this block splits the longer query search phrases into n-grams of size 2
                                        # example- "online safety for children" becomes:
                                        # ['online safety', 'safety for', 'for children']
        query_keyword_s_list = long_keyword_splitter(2, query_keyword)
        keyword_dict_gold_standard_tg[key] = query_keyword_s_list + tg_article_kws
        keyword_dict_yake_plus_queryterm[key] = query_keyword_s_list
    else:
        keyword_dict_gold_standard_tg[key] = [item] + tg_article_kws
        keyword_dict_yake_plus_queryterm[key] = [item]

kw_extractor = yake.KeywordExtractor(n=2)  # it just seems like size 2 produced better results than 1 or 3

# filling yake_tg_keyword_dict:
for key in dict_of_query_terms.keys():
    first_3_char_01 = key[:3]
    title = dict_of_TheGuardian_article_content[first_3_char_01][0]
    body = dict_of_TheGuardian_article_content[first_3_char_01][1]
    text = title + body
    keyword_tupels = kw_extractor.extract_keywords(text)
    list_of_kws = []
    for tupe in keyword_tupels:
        list_of_kws.append(tupe[0])
    keyword_dict_yake_tg[key] = list_of_kws

# combining the kw dicts:
for key in keyword_dict_gold_standard_tg.keys():
    keyword_dict_combined[key] = keyword_dict_gold_standard_tg[key] + keyword_dict_yake_tg[key]

for key in keyword_dict_gold_standard_tg.keys():
    keyword_dict_yake_plus_queryterm[key] = keyword_dict_yake_plus_queryterm[key] + keyword_dict_yake_tg[key]


# ***********************************************************************************************
# this block finds the top keyword-scores per query/query_id
dict_of_top_keyword_scores_per_query = {}  # {'G01.1': [(doc_id, best_kw_score), (doc_id, 2nd_best_kw_score)]}
giant_dict_of_all_the_json_files = read_all_jsons(directory_of_large_json_files_from_elastic_search)
        # {'G01_1': {document01_id_int: ('title', 'body'), document02_id_int: ('title', 'body')},
        #  'G01_2': {document01_id_int: ('title', 'body'), document02_id_int: ('title', 'body')} }
top_100_documents = top_keyword_comparisons(keyword_dict_combined, giant_dict_of_all_the_json_files, 100, kw_extractor)
top_100_documents01 = top_keyword_comparisons(keyword_dict_yake_tg, giant_dict_of_all_the_json_files, 100, kw_extractor)
top_100_documents02 = top_keyword_comparisons(keyword_dict_yake_plus_queryterm, giant_dict_of_all_the_json_files, 100, kw_extractor)
dict_to_txt(top_100_documents, "GoldenYake_top_100")
dict_to_txt(top_100_documents01, "SimpleYake_top_100")
dict_to_txt(top_100_documents02, "SimpleTake_and_query_term_top_100")


# ***********************************************************************************************
# this block of code uses the top 100 results from calling the queries, and then
# gets a baseline score for them. So I know if anything I do is improving the baseline.
base_line_top_100_dict_from_json = read_all_jsons_for_baseline(directory_of_top_100_query_results)
base_line_top_100_dict = {}
for key, dict00 in base_line_top_100_dict_from_json.items():
    this = key
    that = dict00
    reformatted_key = key.replace("_", ".")  # gets the query id in G01.1 format instead of G01_1
    top_100_list = []
    for key00, value00 in dict00.items():
        score = value00[2]
        top_100_list.append((key00, score))
    base_line_top_100_dict[reformatted_key] = top_100_list
dict_to_txt(base_line_top_100_dict, "SimpleText_baseline_top_100_with_query_scores")


# ***********************************************************************************************
# getting the data ready for fine-tuning
qrel_dict = read_all_jsons(directory_of_json_files_from_the_qrel_file)
qrel_training_file = open("Seans_qrel_file_training", "r")
fine_tuning_data_list = []
for line in qrel_training_file:

    # Split the qrel file line into its components (query ID, 0, document ID, and relevance score)
    components = line.split()
    query_id = components[0]
    zero = components[1]  # components[1] is just a bunch of zeros. I don't use this
    document_id = components[2]
    relevance_score = int(components[3])

    # the academic article's title + abstract, aa stands for academic article
    aa_title = qrel_dict[document_id][int(document_id)][0]
    aa_abstract = qrel_dict[document_id][int(document_id)][1]
    aa_text = aa_title + aa_abstract

    # the guardian article title + text
    query_id_first_3_char = query_id[:3]
    TheGuardian_article_title = dict_of_TheGuardian_article_content[query_id_first_3_char][0]
    TheGuardian_article_body = dict_of_TheGuardian_article_content[query_id_first_3_char][1]
    TheGuardian_article_text = TheGuardian_article_title + TheGuardian_article_body

    # formatting the data for training
    if relevance_score == 2:
        relevance_score = 1  # this changes all 2's to 1's for training purposes,
                             # now there are 570 pairs labeled 0 and 458 pairs labeled 1
    data_item = prepare_data_for_fine_tuning(TheGuardian_article_text, aa_text, float(relevance_score))
    fine_tuning_data_list.append(data_item)


# ***********************************************************************************************
# this block of code is to fine tune a sentence bert model
model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
the_model = SentenceTransformer(model_name)
fine_tuning_Dataset = MyDataset(fine_tuning_data_list)  # turning the list of data into a PyTorch 'Dataset' object
train_dataloader = DataLoader(fine_tuning_Dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(the_model)
the_model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=1, warmup_steps=100, output_path="simple_text_fine_tuned_sentence_bert_model_01")
the_model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=2, warmup_steps=100, output_path="simple_text_fine_tuned_sentence_bert_model_02")


# ***********************************************************************************************
# this block reranks the baseline results with the fine-tuned SBERT models
fine_tuned_model_01_epoch = SentenceTransformer("./simple_text_fine_tuned_sentence_bert_model_01")
sbert_top_100_dict_model01 = get_s_bert_top_100_matches(
    base_line_top_100_dict_from_json,
    dict_of_TheGuardian_article_content,
    fine_tuned_model_01_epoch)
dict_to_txt(sbert_top_100_dict_model01, 'sbert_reranking_of_baseline_100_model_01')

fine_tuned_model_02_epoch = SentenceTransformer("./simple_text_fine_tuned_sentence_bert_model_02")
sbert_top_100_dict_model02 = get_s_bert_top_100_matches(
    base_line_top_100_dict_from_json,
    dict_of_TheGuardian_article_content,
    fine_tuned_model_02_epoch)
dict_to_txt(sbert_top_100_dict_model02, 'sbert_reranking_of_baseline_100_model_02')

