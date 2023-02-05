Lab 2 - February 3, 2023
Aimee Haas, Sean Fletcher, SJ Franklin

Summary:

Program traverses through a defined directory path, the_directory_path,
and stores every text file it finds in a list. 

The second step set of functions goes through the list of text files and
extracts the dates, then reformats them into month/day/year.

The third step goes through the list of text files and extracts strings in a
word-number-word pattern, combines the results in a string, and then uses that
string to create a word cloud of the 20 most frequently used words.

A PDF of the word cloud analysis is also included, labeled as
"Lab2 - Step 3 Analysis"

Notes:
- If you already have TLS-Covid19 folder installed, you can
change the directory path through "the_directory_path" variable.
If you do not have TLS-Covid19 folder installed, you can extract it
from the zipped folder included

- WordCloud package requires Visual Studio Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

- WordCloud can be manually downloaded from here for various
configurations:
https://github.com/sulunemre/word_cloud/releases/tag/2
