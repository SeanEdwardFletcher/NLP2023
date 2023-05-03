import csv
import requests
from bs4 import BeautifulSoup
import re


# this function only grabs the URLs for The Guardian
def get_the_gardians_queryIDs_and_urls(csv_file):
    query_id_url_dict = {}
    # url_list = []
    # list_of_topic_ids = []
    counter = 0
    with open(csv_file, 'r') as c_s_v:

        reader = csv.reader(c_s_v, delimiter=';')

        for row in reader:  # each row has 6 strings: topic_id;topic_text;topic_url;query_id;query_text;abstract_url

            if counter == 0:  # this is if-else block is to skip the first row of the CSV file
                counter += 1
                continue
            else:
                # query_id_url_dict[row[0]] = row[2].lower()
                topic_id = row[0]
                # if topic_id in list_of_topic_ids:  # this is so it doesn't grab the same URL twice
                #     continue
                if topic_id[0] == 'T':  # this is so this function ONLY grabs the URLs for The Guardian
                    continue
                else:
                    query_id_url_dict[topic_id] = row[2].lower()

                    # topic_url = row[2].lower()
                    # list_of_topic_ids.append(topic_id)
                    # url_list.append(topic_url)

    return query_id_url_dict


def get_techxplores_urls(csv_file):
    url_list = []
    list_of_topic_ids = []
    counter = 0
    with open(csv_file, 'r') as c_s_v:

        reader = csv.reader(c_s_v, delimiter=';')

        for row in reader:  # each row has 6 strings: topic_id;topic_text;topic_url;query_id;query_text;abstract_url

            if counter == 0:  # this is if-else block is to skip the first row of the CSV file
                counter += 1
                continue
            else:
                topic_id = row[0]
                if topic_id in list_of_topic_ids:  # this is so it doesn't grab the same URL twice
                    continue
                elif topic_id[0] == 'G':  # this is so this function only grabs the URLs for The Guardian
                    continue
                else:
                    topic_url = row[2].lower()
                    list_of_topic_ids.append(topic_id)
                    url_list.append(topic_url)

    return url_list


# this function is built around The Guardian's HTML element:  div id="maincontent"
def get_theguardians_content(url_00):

    # making the call to The Guardian's website
    response = requests.get(url_00)

    # getting the html
    the_text = response.text

    # these two functions use regex to grab an article's keywords and title
    list_of_keywords = get_theguardians_keywords(the_text)
    the_title = get_theguardians_title(the_text)

    # Parse the HTML content of the webpage using Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the div element with the id "maincontent"
    main_content = soup.find('div', {'id': 'maincontent'})

    # Extract the content from the maincontent div element
    the_content = main_content.get_text()

    return the_title, the_content, list_of_keywords


def get_theguardians_title(text):
    # this regex grabs everything after:  "og:title" content="  and before:  "
    # so given this text  "og:title" content="This is the title of the article."
    # it would grab:  This is the title of the article.
    regex = r'(?<="og:title" content=")[^"]+'
    match = re.search(regex, text)
    if match:
        title = match.group(0)
        return title
    else:
        print("sometime broke in the get_theguardians_title() function")


def get_theguardians_keywords(text):
    # this regex grabs everything after:  "keywords":"  and before:  "
    # so given this text "keywords":"keyword01,keyword02,keyword03"
    # it would grab:  keyword01,keyword02,keyword03
    regex = r'(?<="keywords":")[^"]+'
    match = re.search(regex, text)
    if match:
        keywords = match.group(0)
        list_of_keywords = keywords.split(",")
        return list_of_keywords
    else:
        print("sometime broke in the get_theguardians_keywords() function")


def get_techxplores_main_content(url_00):

    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    headers = {"user-agent": USER_AGENT}

    # make the call to the webpage
    response = requests.get(url_00, headers=headers)

    # print(response.content)
    # Parse the HTML content of the webpage using Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the div element with the id "article-main"
    # article_main = soup.find("section", {"class": "article-main"})
    article_main = soup.select_one('div.article-main')
    # Extract the content from the article-main div element
    content = article_main.get_text()

    return content

