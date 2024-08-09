import spacy
import lftk


""" Top 10 linguistic features from LPTR w/ pearson corr to truthful response
1 corrected_adjectives_variation 0.114 -  corr_adj_var
2 root_adjectives_variation 0.114 -  root_adj_var
3 total_number_of_unique_adjectives 0.106 - n_uadj
4 simple_adjectives_variation 0.104 - simp_adj_var
5 average_number_of_adjectives_per_sent 0.103 - a_adj_ps
6 avg_num_of_named_entities_norp_per_word 0.099 - a_n_ent_norp_pw
7 average_number_of_adjectives_per_word 0.098 - a_adj_pw
8 total_number_of_adjectives 0.097 - n_adj
9 corrected_nouns_variation 0.093 - corr_noun_var
10 root_nouns_variation 0.093 - root_noun_var
"""


def extract_features(current_example, features):
    """
    Gets the linguistic features from an example string using lftk
    :param current_example: a string of text
    :param features: which features should be extracted
    :return: features and their scores
    """
    nlp = spacy.load('en_core_web_sm')
    tokenized = nlp(current_example)
    print("tokenized", tokenized)
    # LFTK.customize()  -- for extra params to calculate handcr differently
    LFTK = lftk.Extractor(docs=tokenized)
    extracted_features = LFTK.extract()

    return extracted_features


if __name__ == '__main__':
    features = ['corr_adj_var', 'root_adj_var', 'n_uadj', 'simp_adj_var', 'a_adj_ps', 'a_n_ent_norp_pw', 'a_adj_pw',
            'n_adj', 'corr_noun_var', 'root_noun_var']

    extracted = extract_features("some example text", features=None)
    print(extracted)