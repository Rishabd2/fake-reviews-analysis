"""Pronoun and length features used in the Ott-style EDA."""


def ling_feature(df):
    """Add word, character, and pronoun counts to a preprocessed review frame.

    Expects columns Ori_Review, Clean_Review, and Word_List.
    Label convention matches the original notebooks: 1 = deceptive, -1 = truthful.
    """
    df_new = df.copy()
    words = df_new["Clean_Review"].apply(lambda x: x.split())
    df_new["num_word"] = words.apply(len)
    df_new["num_coreword"] = df_new["Word_List"].apply(len)
    df_new["num_stopword"] = df_new["num_word"] - df_new["num_coreword"]
    df_new["num_char"] = df_new["Ori_Review"].apply(len)
    df_new["char_per_word"] = df_new["num_char"] / df_new["num_word"]

    first_sing = ["i", "me", "my", "mine", "myself"]
    first_pru = ["we", "our", "us", "ours", "ourselves"]
    third = [
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
        "hers",
        "theirs",
        "himself",
        "herself",
        "themselves",
    ]
    df_new["num_first_sing"] = words.apply(lambda x: count_pronoun(x, first_sing))
    df_new["num_first_pru"] = words.apply(lambda x: count_pronoun(x, first_pru))
    df_new["num_third"] = words.apply(lambda x: count_pronoun(x, third))
    return df_new


def count_pronoun(words, pronouns):
    return len([x for x in words if x in pronouns])
