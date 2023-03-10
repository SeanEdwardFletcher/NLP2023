# NLP 2023 Lab 05
# Sean Fletcher and Shea Durgin

import nltk
import numpy as np
import openai

my_key = "your key here"
openai.api_key = my_key


def cosine_similarity(text01, text02):
    resp = openai.Embedding.create(
    input=[text01, text02],
    engine="text-similarity-davinci-001")
    embedding_a = resp['data'][0]['embedding']
    embedding_b = resp['data'][1]['embedding']
    similarity_score = np.dot(embedding_a, embedding_b)
    return similarity_score


def n_gram_overlap(text01, text02):

    words1 = nltk.word_tokenize(text01)
    words2 = nltk.word_tokenize(text02)

    unigrams1 = set(words1)
    unigrams2 = set(words2)
    bigrams1 = set(nltk.bigrams(words1))
    bigrams2 = set(nltk.bigrams(words2))

    unigram_overlap = len(unigrams1.intersection(unigrams2)) / len(unigrams1.union(unigrams2))
    bigram_overlap = len(bigrams1.intersection(bigrams2)) / len(bigrams1.union(bigrams2))

    return unigram_overlap, bigram_overlap


# define function to ask a question and get a response from OpenAI
def ask_openai(question):
    # set up parameters for the OpenAI API request
    prompt = "Q: " + question + "\nA:"
    model = "text-davinci-002"
    temperature = 0.5
    max_tokens = 1024
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    # make the API request and get the response
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=None
    )
    print("Question asked: " + question)
    # extract and return the response text
    return response.choices[0].text.strip()


# this dictionary is of size 10, the keys are the questions' tags and the values are lists of strings.
# there are three strings in the list:
# the first string is the question (title + body),
# the second is the accepted answer from stack exchange
# the third is the chat GPT generated response
the_questions_and_answers_dict = {
    "syntax": [
        """What is the most commonly accepted synonym or synonymous phrase in linguistics for "wh-question"? The term 
    "wh-question" seems transparent enough for English speakers, but reeks of English language chauvinism. I have 
    heard such questions referred to as "information questions," "content questions, and "question-word questions," 
    but I don't know what synonym or synonymous phrase for "wh-question" is most commonly used among professional 
    linguists these days. I haven't been able to find an answer to my question. Since my hobby is writing reference 
    grammars for imaginary languages, an answer to this question will be useful for me.""",

        """“Content question” is common. I’ve seen that used in typologically oriented grammars of languages from all 
        over the world. I’ve also seen non-polarity question, though I like that option less.""",
        """The most common synonym for "wh-question" is probably "interrogative." """],

    "phonology": [
        """What are near-minimal pairs? What are near-minimal pairs? How are they different from minimal pairs?""",
        """Sure. English "hit" and "hot" are near-minimal pairs, differing only in one phoneme, but the /h/ is realized 
        differently because of that (more like [ç] in the first one). It's not always possible to find a true minimal 
        pair to distinguish phonemes, so near-minimal pairs can sometimes be necessary. You just need to make sure that 
        whatever differs between them isn't likely to be causing allophony.""",
        """Near-minimal pairs are two words that differ by only one phonological feature and are therefore similar in 
        sound. Minimal pairs are two words that differ by only one phoneme and are therefore identical in sound."""],

    "Phonetics": [
        """is schwa a phoneme in English? or is it simply an unstressed allophone of unstressed lax vowels? I'm curious 
    because I've heard some people claim that [ə] is not a phoneme and it is just a reduced allophone of all the 
    unstressed vowels in English.""",
        """First, "allophone" is the wrong term to use for many-to-one mappings between phonemes and surface realizations. Allophonic rules are not neutralizing, so final devoicing in German (for example) is not an allophonic rule. Aspiration in Engilsh is, because only /t/ becomes [tʰ].
Second, just to be clear, we say that the voiceless alveolar stop in English is the phoneme /t/ and not the phoneme /tʰ/ because the rule deriving [tʰ] from /t/ is simpler than the converse rule deriving [t] from /tʰ/. Or, so it has been thought – but it turns out that if you state the rule in terms of stress-foot position ("deaspirate foot-medially") you not only can state deaspiration as easily as you can state aspiration – plus, you can also explain why /h/ deletes in words like "prohibition". So the analogous question is, is it easier to derive [ʌ] from /ə/, or is it easier to derive [ə] from /ʌ/? There is no difference in simplicity: you get schwa when unstressed and caret when stressed. So it seems to be a coin toss.
But third: lots of (unstressed) vowels become schwa in English, as one can see from alternation like monotone ~ monotony [ˈmɔnətoʊn, məˈnɔtəni] though in some dialects, my [ɔ] is [a]; telegraph ! telegraphy [ˈtɛləgræf, təlɛgrəfi]. There is a many-to-one mapping between all sorts of vowels and schwa: so this is not an allophonic relation. Not all rule-governed relations in phonology are allophonic.
The conclusion therefore would be that every instance of schwa in English can be derived from some other vowels when unstressed. Sometimes you can tell that it comes from /ɪ/ or /ɛ/ given stress alternations as in telephone ~ telephonic. Sometimes you cannot tell: the first and last syllables of vanilla, first syllables of collapse, giraffe. The decision to say that the first vowel of vanilla must be underlyingly some vowel other than schwa is based on an unsupported claim that English underlyingly has no instances of schwa. That vowel could be ɪ, or ʌ, or ɛ, or æ...
The motivation for purging schwa from underlying forms is the idea that there is an intolerable acquisition burden to learning 12 phonemes compared to 11 phonemes. However, the rationale is countered by the heavier burden on acquisition that comes from having to make an arbitrary decision regarding the underlying form of many words. Whereas, if the default assumption is that the underlying form is the same as the surface form unless there is good evidence saying otherwise (and "saves a phoneme" is not good enough evidence), then schwa can be a phoneme of English, meaning that it is present in underlying forms.
""",
        """There is some debate on this topic, but most linguists agree that schwa is a phoneme in English. This is because it is a distinct sound that can change the meaning of a word (e.g. "a" vs. "at"), and it is also represented by a separate letter in the alphabet (the letter "e")."""],

    "terminology": [
        """Is there a term for "if" statements not intended as conditions, like "if you want" I might say, "There is food 
    in the fridge, if you want, Fred." I do not mean that food in the fridge will only exist if Fred wants it to exist. 
    I mean, "There is food in the fridge, [which I mention so that you know about it] if you want [to get yourself 
    any]." Since I first noticed this sort of phrase, I notice variation on it all the time. Is there a term for this 
    sort of "condition"? """,

        """It's called Biscuit Conditionals. Several sources say the term was coined in Austin (1956), though I haven't 
        checked the original paper. Austin, J. L. (1956) Ifs and cans. Proceedings of the British Academy 42, 109–132""",

        """I think you are describing a conditional statement that is not actually a conditional statement. The "if" in 
        "if you want" is not intended to express a condition. It is intended to introduce the idea that the speaker is 
        giving the listener a choice."""],

    "historic": [
        """How did the generic masculine emerge? In an essay for school I recently claimed the generic masculine was 
        caused by sexism, but my teacher complained that I hadn't given a reason for this. Assuming my hypothesis is 
        correct, how did this develop (I'm not asking about a gender system or sexism – the web has a lot on these – 
        but on the generic masculine)? At least to me (who has always known of its existence) it's obvious that the 
        male form would be also linguistically preferred, but I can't come up with any mechanism for this. When I 
        tried to search the web for it, I only found that prescriptivism (together with sexism of course) has 
        significantly accelerated it in the English language once it already existed somewhat, but not how it started 
        in English nor how it worked in any language that actually has a real genus system (like my native language 
        German, or Latin).""",
        """In many Indo-European languages, like Latin, the masculine is less "marked" than the feminine, meaning that 
        it's the more basic or fundamental form: the one you use by default unless there's a reason to do otherwise. 
        While sexism might play a role in this (certainly the ancient Romans weren't particularly feminist), there's 
        also a more mundane historical reason. The feminine gender seems to have been a later development in the 
        history of Proto-Indo-European, which made it more marked than the masculine or the neuter—in other words, 
        the three genders were originally "animate", "inanimate", and "this special new marking for 
        specifically-feminine things". If something wasn't specifically feminine, it didn't get the special new 
        marking. This seems to have led to the convention that was inherited by Latin, that groups of people and 
        generic individuals used the masculine gender.
Of course, this was thousands of years ago. The generic masculine in modern English is a recent development, as you 
noted: English used the non-gendered "they" for groups of people and hypothetical/non-specific individuals until 
prescriptive efforts arose to make it more like Latin. (You can find lots of traces of these prescriptive efforts in 
modern English: "don't split infinitives" and "don't strand prepositions" are similar rules imposed to make English 
more like Latin, which are still taught in schools but most people don't really follow.)
Other languages may have the convention for other reasons. In Proto-Afro-Asiatic, there seems to have been a two-way 
masculine/feminine gender distinction—but when using an adjective generically, this changed to an animate/inanimate 
(or sometimes concrete/abstract) split. So while "good man" would be masculine and "good woman" feminine, 
"some good person" would also be masculine, and "some good thing" or "quality of goodness" would be feminine. 
This is the system that appears in Ancient Egyptian and Akkadian, and likely also in some modern Semitic languages 
(though I don't know any of them in enough detail to say for sure). This leads to a "generic masculine" convention, 
but for fairly arbitrary reasons: repurposing their existing morphology to distinguish between "good person" and 
"quality of goodness".
Tl;dr this happened for different reasons in different languages; sexism may well have had something to do with it, 
but there are other (often-arbitrary) historical forces in play.
""",
        """The generic masculine is a form of linguistic sexism that results in the use of male-oriented language to 
        refer to both sexes. This can take the form of using masculine pronouns (he, him, his) to refer to both men 
        and women, or using masculine terms (man, boy, gentleman) to refer to both sexes.

        The generic masculine has been shown to have a number of negative effects. First, it can make women feel 
        invisible or unimportant. Second, it can reinforce the idea that men are the default or "normal" human beings, 
        while women are somehow different or other. Finally, it can lead people to make inaccurate assumptions about 
        who is being referred to when a masculine pronoun or term is used. 

        The generic masculine is thought to have arisen out of a combination of sexism and linguistic convention. In 
        many languages, the masculine form is the default form when referring to people of unknown or unspecified 
        gender. This is likely due to the fact that, historically, most languages have been patriarchal, and thus 
        have favored the masculine form. 

        The generic masculine has been the subject of much debate in recent years, as more and more people have 
        become aware of its existence and effects. Some people argue that it should be eradicated from the language 
        in order to promote gender equality. Others argue that it can be used in a non-sexist way, and that its 
        elimination would be a form of linguistic censorship."""],

    "etymology": [
        """History of perfect tenses I am thinking about the history of the verb "have". Why is the verb "have" used as an 
    auxiliary verb in the perfect tenses? When did it start to be used that way?""",
        """According to the OED (have, sense VI):
The have-perfect in English apparently arose as a reanalysis of uses such as I have my work done ‘I have my work in a done or finished condition’; the complement done was reinterpreted as part of the verb phrase, a process which was reinforced by a lack of fixed word order and the possible transposition of object and participle, i.e. I have done my work. This development appears to have largely taken place before the written record. Even in early Old English, in the majority of examples with transitive verbs the past participle is not inflected to agree with the object. Despite occasional ambiguity, there are few Old English examples in which the past participle must be regarded as a complement rather than as part of a perfect construction.
In Old English, the have-perfect is not only established with transitive verbs, but also with intransitive verbs expressing action or occurrence, while the perfect of intransitive verbs expressing change of state or position is usually formed with be. From Middle English onwards the perfect with have gradually becomes more common in these verbs, and is the predominant form by the early 19th cent., except in contexts where the focus is on resultant state (for example, she is gone is still typically used to express state, while she has gone expresses action; such usage is now, however, quite limited). In early Middle English the have-perfect also extends to verbs denoting ongoing states or conditions, and to the verb to be.
This "have-perfective" is one of the most significant features of the Standard Average European sprachbund—in other words, it's a feature that's shared by a lot of not-necessarily-related languages in a particular area.
There's some disagreement about where in the sprachbund it originated, but personally, I like the theory that it arose in Vulgar Latin. Like the OED described, it seems to involve re-analysis of sentences like (litterās scriptās) habeō "I have (written letters)" as litterās (scriptās habeō) "I (have written) letters". The ancestor of English "have" (and German haben and such) was then an easy equivalent for the etymologically-unrelated Latin habēre.
Latin was written much earlier than Old English, and in the Latin record we can see this construction evolving, from marginal uses in Classical texts (there's one attestation in Vitruvius that scholars disagree about how to analyze) to a variety of Vulgar attestations to becoming the standard in Romance. In Classical Latin, a past aoristic verb ("I wrote") and a present perfect verb ("I have written") look identical, so there was plenty of reason for Vulgar Latin to develop a way to distinguish the two, leading to the reanalysis mentioned above.
It's also possible that this development started in Germanic or somewhere else and spread from there to Latin. Since we don't have early written records of these other languages, this is a difficult claim to falsify—as the OED mentions, the oldest written forms of English show this construction. But, the earliest documentation of it seems to be in Vulgar Latin, and I find it very easy to believe it spread from Romance into Germanic.
""", """The use of "have" as an auxiliary verb in the perfect tenses is a relatively recent development in the history of the English language. It began to be used in this way in the late Middle English period (around the 14th century), and became more common in the early Modern English period (around the 16th century).
"""],

    "Grammar": [
        """ About the location of 'đươc', can it be placed both before a verb and before a noun? In my Vietnamese 
    self-learning book, I saw this sentence: Rất vui đươc gặp cô. I'm very pleased to meet you. (To an elderly lady) 
    So I thought đươc is used like đươc + verb (đươc gặp = can meet). But later, I found this sentence: May mà tôi gặp 
    lại đươc cô. Luckily I can see you again. (To an elderly lady) Does this mean đươc expresses possibility and can 
    be placed before a verb and even before a noun? And in this second case was it placed before cô because it 
    emphasises cô? Why didn't they say this: May mà tôi đươc gặp lại cô. Would that be wrong? """,
        """The word được (note the tone) is both a transitive action verb, with a principal meaning of "to win" or "get", and extended to "attain" and then as a modal verb "must". Compare English "to get to do something". These belong before verbs, and can even have nouns after them.
Bản thân tôi cũng mong muốn con mình được đi học.
I also want my children to get (/ to have the opportunity) to go to school.
It is also a particle that expresses ability and possibility, following the verb.
Its origins lie in an early borrowing (non-'Classical' Sino-Vietnamese) of the Chinese 得, and its uses have strong parallels in Cantonese and Mandarin.
""",
        """Yes, được can be placed before a verb or a noun.
When placed before a verb, it expresses the possibility of doing that verb.
When placed before a noun, it expresses the possibility of having that noun.
The second sentence is saying that the speaker is glad that they have the opportunity to see the person again. The speaker could have said "May mà tôi đươc gặp lại cô" which would have the same meaning, but placing đươc before cô emphasizes the person."""],

    "IPA": [
        """Difference between Cantonese /gw/ and Mandarin /gu/? As a native speaker of both languages, Cantonese /gw/ like 
    in 過gwo3 and Mandarin /gu/ like in 过guo4 sounds the same, but I've checked that the Cantonese one is [kʷɔː] while 
    the Mandarin one is [kwo], is there really a difference?""",
        """In an extremely narrow transcription:
[kʷ] is a single sound, a type of [k] with rounded lips. The lips are rounded at the same time as the occlusion starts, and they're unrounded as soon as the occlusion ends (plosive burst).
[kw] is a sequence of two sounds. First you get a [k], with unrounded lips; you'll get the occlusion and the burst. Then you get the [w], where the lips are rounded and the tongue gets close to the velum without touching it, as typical for approximants.
In practice... well. Speech does not segment so neatly. More often than not, you'll hear speakers using [kʷ], [kʷw] and [kw] interchangeably, sometimes even for the same word in different utterances. So when you're transcribing a language, there's a lot of room to transcribe all those sounds as [kʷ] or [kw].
Often the decision between one or another will be personal, and up to the author. Sometimes however you'll get people transcribing it as [kʷ] because, for the sake of phonotactics, it behaves like a single unit; while [kw] would behave as two units.
Sorry if this does not directly answer your question, as I don't speak either language. But hope that it helps anyway.
""",
        """There is a difference between Cantonese /gw/ and Mandarin /gu/. The Cantonese /gw/ is pronounced [kʷɔː], while the Mandarin /gu/ is pronounced [kwo]. The difference is that the Cantonese /gw/ has a rounded vowel, while the Mandarin /gu/ does not.
"""],

    "Corpora": [
        """Large parallel corpus of English and Farsi text hope you are doing well. I am looking for a large parallel 
    corpus of English and Farsi text, professionally written or edited. I would be more than happy to have your 
    suggestions. Thank you all. """,
        """With the criteria you give (large, parallel, professionally written or edited) there is probably just one such corpus: The Multilingual Bible Parallel Corpus, for a more formal reference see here. It has its limitations: The texts belong to a very specific register and genre, and they lack a lot of modern terminology (no airplanes nor bicycles occur there).
An alternative may be the JW corpus created from a crawl of the website of Jehova's Witnesses. Again, this corpus contains religious material, but its source is written in modern English and it contains a higher proportion of modern vocabulary. I cannot assess the degree of professionality that has gone in the Farsi texts collected from that site.""",
        """There are a few different options for finding parallel corpora:
    
    The Universal Dependencies project has a number of different language pairs available.
    The European Union has a parallel corpus available for a number of languages.
    The National Translation Initiative has a number of different language pairs available."""],

    "Sociolinguistics": [
        """Is there a linguistic term for apologetic prefacing? I was editing a question on Stack Overflow. Like so many 
    questions it started with an apologetic or diminishing preface: I am genuinely sorry if this is seen as simple but 
    I am new to coding in general but here goes nothing. I will just post the question here but Ill try and explain it 
    myself as well. For SO, I can understand. It can be very harsh towards new people and phrasing like the above makes 
    a downvote much more likely as people want to simply see questions, code, error logs ... and answers. But it does 
    make me wonder if there is term, perhaps in sociolinguistics, for this sort of "please don't hurt or reject me" 
    phrasing that's so common.""",
        """This is known in sociolinguistics and pragmatics as a hedge — a way to soften the certainty and definiteness of an utterance by way of indirection.
In your example, an utterance implying "I am a total noob, here's my question." is softened to "If it happens that I am a total noob, then I'm terribly sorry. Here's my question anyway."
""",
        """I'm not sure if there's a term for this specific phenomenon, but it's a type of hedging. From the Cambridge Grammar of 
    the English Language:
<blockquote>
<p><strong>Hedges</strong> are words which are used to make the contribution which they
  introduce less definite or less committed. They are typically used to
  convey the speaker's attitude to the information conveyed: for
  example, to express doubt, or to make the information conveyed sound
  less certain. The notion of hedge is often illustrated with the
  following examples:</p>
<p>I <strong>think</strong> that ...</p>
<p>I <strong>guess</strong> that ...</p>
<p>I'm <strong>afraid</strong> that ...</p>
<p>It <strong>seems</strong> that ...</p>
<p>I'm <strong>not sure</strong> that ...</p>
<p>The <strong>fact is</strong> that ...</p>
<p><strong>Basically</strong>, ...</p>
<p>I <strong>suppose</strong> so.</p>
<p>I <strong>daresay</strong> so.</p>
<p>I <strong>imagine</strong> so.</p>
<p>I <strong>expect</strong> so.</p>
</blockquote>"""],
}

# the following two "manual_analysis" dictionaries are formatted so that the keys are the question tags
# the values are each a list of 2 lists
# the first inner-list is the relevance and correctness score for the stack exchange accepted answer, 0 or 1
# the second inner-list is the relevance and correctness score for the chat gpt generated answer, 0 or 1
manual_analysis_dict_shea = {
    "syntax":           [[1, 1],   [1, 0]],
    "phonology":        [[1, 1],   [1, 1]],
    "phonetics":        [[1, "x"], [1, "x"]],
    "terminology":      [[1, 1],   [1, 1]],
    "historic":         [[1, 1],   [1, 0]],
    "etymology":        [[1, 1],   [1, 0]],
    "grammar":          [[1, 1],   [1, 1]],
    "IPA":              [[1, 1],   [1, 1]],
    "corpora":          [[1, 1],   [1, 0]],
    "sociolinguistics": [[1, 1],   [1, 1]]
}
manual_analysis_dict_sean = {
    "syntax":           [[1, 1],   [1, 0]],
    "phonology":        [[1, 1],   [1, 0]],
    "phonetics":        [[1, "x"], [1, 0]],
    "terminology":      [[1, 1],   [0, 0]],
    "historic":         [[1, 1],   [1, 0]],
    "etymology":        [[1, 1],   [1, 0]],
    "grammar":          [[1, 1],   [1, 1]],
    "IPA":              [[1, 1],   [1, 1]],
    "corpora":          [[1, 1],   [1, 0]],
    "sociolinguistics": [[1, 1],   [1, 1]]
}


# function calls
automatic_analysis_dict = {}

# for key in the_questions_and_answers_dict.keys():
#     text01 = the_questions_and_answers_dict[key][1]
#     text02 = the_questions_and_answers_dict[key][2]
#     automatic_analysis_dict[key] = [cosine_similarity(text01, text02)]
#
#     unigram_overlap, bigram_overlap = n_gram_overlap(text01, text02)
#
#     automatic_analysis_dict[key].append(unigram_overlap)
#     automatic_analysis_dict[key].append(bigram_overlap)

# for key in automatic_analysis_dict.keys():
#     print("*************")
#     print(key)
#     print("-")
#     print(automatic_analysis_dict[key][0])
#     print(automatic_analysis_dict[key][1])
#     print(automatic_analysis_dict[key][2])
#     print("*************")


def get_the_auto_averages(input_dict):
    average_cosine_sim = 0
    average_unigram_overlap = 0
    average_bigram_overlap = 0
    for key0 in input_dict.keys():
        average_cosine_sim += input_dict[key0][0]
        average_unigram_overlap += input_dict[key0][1]
        average_bigram_overlap += input_dict[key0][2]

    return average_cosine_sim/10, average_unigram_overlap/10, average_bigram_overlap/10


# cosine_sim_av, uni_overlap_av, bi_overlap_av = get_the_auto_averages(automatic_analysis_dict)
# print("---")
# print("cosine similarity average:")
# print(cosine_sim_av)
# print("---")
# print("unigram overlap average:")
# print(uni_overlap_av)
# print("---")
# print("bigram overlap average:")
# print(bi_overlap_av)
# print("---")


def get_the_manual_averages(input_dict01, input_dict02):
    se_relevance_av = 0
    se_correct_av = 0
    gpt_relevance_av = 0
    gpt_correct_av = 0

    for key00 in input_dict01.keys():
        if key00 != "phonetics":
            se_relevance_av += input_dict01[key00][0][0]
            se_correct_av += input_dict01[key00][0][1]
            gpt_relevance_av += input_dict01[key00][1][0]
            gpt_correct_av += input_dict01[key00][1][1]

            se_relevance_av += input_dict02[key00][0][0]
            se_correct_av += input_dict02[key00][0][1]
            gpt_relevance_av += input_dict02[key00][1][0]
            gpt_correct_av += input_dict02[key00][1][1]

    return se_relevance_av/18, se_correct_av/18, gpt_relevance_av/18, gpt_correct_av/18


# stack_ex_relevance, stack_ex_correctness, gpt_relevance, gpt_correctness = get_the_manual_averages(
#     manual_analysis_dict_shea,
#     manual_analysis_dict_sean)
# print(stack_ex_relevance)
# print(stack_ex_correctness)
# print(gpt_relevance)
# print(gpt_correctness)


