from pyspan.json_to_txt import re_sent_end, get_sentences

def test_get_sentences():
    sentence1 = "This is a test sentence."
    assert get_sentences(sentence1, mode = "speech") == [sentence1]
    assert get_sentences(sentence1, mode = "sentences") == [sentence1]

    sentence2 = "This is a test sentence. So is this."
    assert get_sentences(sentence2, mode = "speech") == [sentence2]
    assert get_sentences(sentence2, mode = "sentences") == \
        ["This is a test sentence.", "So is this."]

    sentence3 = "Is this a test sentence? I guess so."
    assert get_sentences(sentence3, mode = "speech") == [sentence3]
    assert get_sentences(sentence3, mode = "sentences") == \
        ["Is this a test sentence?", "I guess so."]

    sentence4 = "This might be a test sentence...what do you think?"
    assert get_sentences(sentence4, mode = "speech") == [sentence4]
    assert get_sentences(sentence4, mode = "sentences") == [sentence4]

    ## I don't see a way around "edge cases" like these.
    #sentence5 = "Mr. Jones lives on 5th Ave. Mrs. Smith visited Dr. Doe, who works on 4th St., next to the grocery store."
    #assert get_sentences(sentence5, mode = "speech") == [sentence5]
    #assert get_sentences(sentence5, mode = "sentences") == \
    #    ["Mr. Jones lives on 5th Ave.", 
    #     "Mrs. Smith visited Dr. Doe, who works on 4th St., next to the grocery store."]

    sentence6 = "The House votes on H. Res. 1 today. I ask unanimous consent..."
    assert get_sentences(sentence6, mode = "speech") == [sentence6]
    assert get_sentences(sentence6, mode = "sentences") == \
        ["The House votes on H. Res. 1 today.", "I ask unanimous consent..."]
 
    sentence7 = "This is 1.2 test sentences."
    assert get_sentences(sentence7, mode = "speech") == [sentence7]
    assert get_sentences(sentence7, mode = "sentences") == [sentence7]
 
    sentence8 = "LT. Bob won an award! Now I'll say a new sentence."
    assert get_sentences(sentence8, mode = "speech") == [sentence8]
    assert get_sentences(sentence8, mode = "sentences") == \
        ["LT. Bob won an award!", "Now I'll say a new sentence."]

    sentence9 = "This is a sentence about salt. This is a sentence about pepper."
    assert get_sentences(sentence9, mode = "speech") == [sentence9]
    assert get_sentences(sentence9, mode = "sentences") == \
        ["This is a sentence about salt.", "This is a sentence about pepper."]

    sentence10 = "This is a sentence about SALT. This is a sentence about PEPPER."
    assert get_sentences(sentence10, mode = "speech") == [sentence10]
    assert get_sentences(sentence10, mode = "sentences") == \
        ["This is a sentence about SALT.", "This is a sentence about PEPPER."]

    sentence11 = "Can you believe this is a sentence?! What a world..."
    assert get_sentences(sentence11, mode = "speech") == [sentence11]
    assert get_sentences(sentence11, mode = "sentences") == \
        ["Can you believe this is a sentence?!", "What a world..."]

    sentence12 = "Mr. Jones lives on 5th Ave., next to the pharmacy. Mrs. Smith visited Dr. Doe, who works on 4th St., next to the grocery store."
    assert get_sentences(sentence12, mode = "speech") == [sentence12]
    assert get_sentences(sentence12, mode = "sentences") == \
        ["Mr. Jones lives on 5th Ave., next to the pharmacy.", 
         "Mrs. Smith visited Dr. Doe, who works on 4th St., next to the grocery store."]
