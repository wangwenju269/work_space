import jieba
import math
import string
import unicodedata
import re
import json
import os
import multiprocessing
from collections import Counter


def unify_format(text):
    # This rule needs to detect Chinese character'【】', therefore done before data format unification
    # rule 6: The fraction of full brackets "[]" in words should be <= 0.1
    normalized_content = unicodedata.normalize('NFKC', text)
    bracketed_texts = re.findall(r'【(.*?)】', normalized_content)
    # Calculate the total number of characters inside brackets
    characters_inside_brackets = sum(len(text) for text in bracketed_texts)
    # Calculate the total number of characters in the text
    total_characters = len(normalized_content)
    # Calculate the ratio of characters inside brackets to the total characters
    ratio = characters_inside_brackets / total_characters if total_characters > 0 else 0
    # Check if the fraction is less than or equal to 0.1
    if ratio > 0.1:
        return False, None, {'bracket ratio': ratio}

    # Construct the mapping table from full-angle symbols to half-angle symbols
    halfwidth_symbols = ".!\"\"#$%&''()*+,-/:;<=>?@[\\]^_`{|}~"
    fullwidth_symbols = "。！“”＃＄％＆‘’（）＊＋，－／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"
    translation_table = str.maketrans(fullwidth_symbols, halfwidth_symbols)
    # translate to realize conversion
    fullwidth_text = text.translate(translation_table)
    return fullwidth_text

def url_filter(text):
    # remove links in the text
    # First use regular expression to remove common links
    url_pattern_1 = r"https?://[^\s,，\u4e00-\u9fa5]*|www\.[^\s,，\u4e00-\u9fa5]*"
    first_pass_result = re.sub(url_pattern_1, "", text)
    # Second regular expression matching domain names containing three dots, removing word boundary restrictions and considering the possibility of dots at the end
    url_pattern_2 = r"[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\.?"
    second_pass_result = re.sub(url_pattern_2, "", first_pass_result)
    # Third regular expression to remove urls not prefixed with http/https/www, which will match urls like 'domain.com'
    url_pattern_3 = r"(?<![\u4e00-\u9fa5])(\b[a-zA-Z0-9.-]+\.[a-z]{2,6}\b)"
    third_pass_result = re.sub(url_pattern_3, "", second_pass_result)
    return third_pass_result


def self_defined_rules(text):
    """ function returns True if the sample complies with self-defined filtering rules """
    sentences = re.split(r'[.!…?;"]+', text)  # Using regular expression to split the document into sentences
    normalized_content = unicodedata.normalize('NFKC', text)

    # rule 1: Number of sentences in a document should be > 1
    # Filter out any empty strings that might result from the split
    sentences = [sentence for sentence in sentences if sentence.strip()]
    # Check if the number of sentences is greater than 1
    if len(sentences) <= 1:
        return False, {'Number of senteces less than 1': len(sentences)}

    # rule 2: The number of words in the content after normalisation must be between 50 and 10'000
    # Normalize the content using NFKC normalization
    # Count the number of 'words' (here considering each character as a word)
    word_count = len(normalized_content)
    # Check if the word count is between 50 and 10,000
    if word_count < 50 or word_count > 10000:
        return False, {'Word count less than 50 or larger than 10000': word_count}

    # rule 3: The mean length of words in the content after normalisation must be between 1.3 and 10
    # remove punctuations in text
    cleaned_text = re.sub(r'[^\w\s]', '', normalized_content)
    # using jieba to tokenize words
    words = [word for word in jieba.cut(cleaned_text) if word.strip()]
    # Calculate the mean length of the words
    mean_length = sum(len(word) for word in words) / len(words) if words else 0
    # Check if the mean length is within the desired range
    if mean_length < 1.3 or mean_length > 10:
        return False, {'Mean word length less than 1.3 or larger than 10': mean_length}

    # rule 4: The fraction of nonconsecutive hashtags in words must be <= 0.1
    hashes_count = len(re.findall(r'#+', text))
    nonconsecutive_hash_ratio = hashes_count / len(text)
    if nonconsecutive_hash_ratio > 0.1:
        return False, {'Hash tag ratios exceeding 0.1': nonconsecutive_hash_ratio}

    # rule 5: The fraction of nonconsecutive ellipsis in words must be <= 0.1
    # regular expression to match ellipsis and dots
    ellipsis_pattern1 = re.compile(r'…+')
    ellipsis_pattern2 = re.compile(r'\.{2,}')
    # count matches
    ellipsis_count = len(re.findall(ellipsis_pattern1, text)) + len(re.findall(ellipsis_pattern2, text))
    if ellipsis_count / len(text) > 0.1:
        return False, {'Ellipsis ratios exceeding 0.1': ellipsis_count / len(text)}

    # rule 6: The fraction of full brackets "【】" in words should be <= 0.1
    # moved to function unify_format()

    # rule 7: The fraction of digital words over non-punctuation words should be <= 0.3
    # Remove punctuation from the content
    no_punct_content = re.sub(r'[^\w\s]', '', normalized_content)
    # In Chinese, considering each character as a 'word' might be more appropriate
    words = list(no_punct_content)
    # Count the number of words that contain digits
    digital_word_count = sum(1 for word in words if any(char.isdigit() for char in word))
    # Calculate the fraction of words with digits over total words (without punctuation)
    total_word_count = len(words)
    fraction = digital_word_count / total_word_count if total_word_count > 0 else 0
    # Check if the fraction is greater than 0.3
    if fraction > 0.3:
        return False, {'Digit over non_punctuation words': fraction}

    # rule 8: The fraction of lines ends with "readmore" etc. should be <= 0.3
    readmore_variants = ["readmore", "展开", "更多", "。。。"]  # Add more variants if needed
    # Split the content into lines
    lines = normalized_content.splitlines()
    # Count the number of lines ending with any of the readmore variants
    readmore_count = sum(1 for line in lines if any(line.endswith(variant) for variant in readmore_variants))
    # Calculate the fraction of lines ending with any of the readmore variants
    fraction = readmore_count / len(lines) if lines else 0
    if fraction > 0.3:
        return False, {'Lines ending with readmore etc.': fraction}

    # rule 9: The fraction of lines starting with a bullet point must be <= 0.9
    # Split the content into lines
    lines = normalized_content.splitlines()
    # Define common bullet point characters
    bullet_points = ['•', '●', '○', '■', '□', '▪', '▫', '※', '·', '➢']
    utf_code = ['\u2022', '\u2023', '\u25B6', '\u25C0', '\u25E6', '\u25A0', '\u25A1', '\u25AA', '\u25AB', '\u2013']
    bullet_points += utf_code
    # Count the number of lines starting with a bullet point
    bullet_point_count = sum(
        1 for line in lines if line and any(line.startswith(bullet) for bullet in bullet_points))
    # Calculate the fraction of lines starting with a bullet point
    fraction = bullet_point_count / len(lines) if lines else 0
    # Check if the fraction is less than or equal to 0.9
    if fraction > 0.9:
        return False, {'Line starting with bullet point': fraction}

    # rule 10: The fraction of punctuations in words must be > 0
    # Chinese text doesn't use spaces to separate words like in English
    # So we consider each character as a 'word' for the purpose of this calculation
    total_chars = len(normalized_content)
    # Define a set of common Chinese punctuations along with English punctuations
    chinese_punctuations = "…"
    all_punctuations = chinese_punctuations + string.punctuation
    # Count the number of punctuation characters
    punctuation_count = sum(1 for char in normalized_content if char in all_punctuations)
    # Calculate the fraction of punctuation characters in the text
    fraction = punctuation_count / total_chars if total_chars > 0 else 0
    # Check if the fraction is greater than 0
    if fraction <= 0:
        return False, {'Punctuation ratio': fraction}

    # rule 11: The fraction of unique words in the content must be > 0.1
    # In Chinese, words are not separated by spaces, so a more advanced method like jieba can be used for segmentation
    # For simplicity, let's consider each character as a word in this example
    words = list(normalized_content)
    # Calculate the number of unique words
    unique_words = set(words)
    # Calculate the fraction of unique words in the content
    fraction_unique = len(unique_words) / len(words) if words else 0
    # Check if the fraction of unique words is greater than 0.1
    if fraction_unique <= 0.1:
        return False, {'Unique words fraction': fraction_unique}

    # rule 12: The entropy of the unigram distribution of the content must be >= 3
    # In Chinese, consider each character as a unigram
    unigrams = list(normalized_content)
    # Count the frequency of each unigram
    unigram_counts = Counter(unigrams)
    total_unigrams = sum(unigram_counts.values())
    # Calculate the entropy of the unigram distribution
    entropy = -sum((count / total_unigrams) * math.log(count / total_unigrams) for count in unigram_counts.values())
    if entropy < 3:
        return False, {'Entropy of unigram distribution': entropy}

    return True, None

def ccnet_rules(text,model):
    """ function returns True if the sample complies with CCNet filtering rules """
    # rule 1: Score of the language identification model must be > 0.4
    # need to move load model to outside of the function to avoid loading model every time
    # model = fasttext.load_model(model_path)
    text = text.replace('\n', '')
    predict = model.predict(text, k=1)
    score = predict[1][0]
    if score < 0.4:
        return False, {'fasttext score': score}
    return True, None


def duplicates_rules(text):
    """ function returns True if the sample complies with duplicates filtering rules """
    total_character_count = len(text)
    if total_character_count == 0:
        return False, {'total character count': total_character_count}

    # rule 1: The fraction of characters in the top word 2gram must be < 0.20
    top2gram_char = count_ngram_duplicates(text, 2)
    if top2gram_char / total_character_count > 0.20:
        return False, {'characters in the top word 2gram': top2gram_char / total_character_count}

    # rule 2: The fraction of characters in the top word 3gram must be < 0.18
    top3gram_char = count_ngram_duplicates(text, 3)
    if top3gram_char / total_character_count > 0.18:
        return False, {'characters in the top word 3gram': top3gram_char / total_character_count}

    # rule 3: The fraction of characters in the top word 4gram must be < 0.16
    top4gram_char = count_ngram_duplicates(text, 4)
    if top4gram_char / total_character_count > 0.16:
        return False, {'characters in the top word 4gram': top4gram_char / total_character_count}

    # rule 4: The fraction of characters in duplicate word 5grams must be < 0.60
    dup5gram_char = count_ngram_duplicates(text, 5)
    if dup5gram_char / total_character_count > 0.6:
        return False, {'characters in duplicate word 5grams': dup5gram_char / total_character_count}

    # rule 5: The fraction of characters in duplicate word 6grams must be < 0.60
    dup6gram_char = count_ngram_duplicates(text, 6)
    if dup6gram_char / total_character_count > 0.6:
        return False, {'characters in duplicate word 6grams': dup6gram_char / total_character_count}

    # rule 6: The fraction of characters in duplicate word 7grams must be < 0.60
    dup7gram_char = count_ngram_duplicates(text, 7)
    if dup7gram_char / total_character_count > 0.6:
        return False, {'characters in duplicate word 7grams': dup7gram_char / total_character_count}

    # rule 7: The fraction of characters in duplicate word 8grams must be < 0.60
    dup8gram_char = count_ngram_duplicates(text, 8)
    if dup8gram_char / total_character_count > 0.6:
        return False, {'characters in duplicate word 8grams': dup8gram_char / total_character_count}

    # rule 8: The fraction of characters in duplicate word 9grams must be < 0.60
    dup9gram_char = count_ngram_duplicates(text, 9)
    if dup9gram_char / total_character_count > 0.6:
        return False, {'characters in duplicate word 9grams': dup9gram_char / total_character_count}

    # rule 9: The fraction of characters in duplicate word 10grams must be < 0.60
    dup10gram_char = count_ngram_duplicates(text, 10)
    if dup10gram_char / total_character_count > 0.6:
        return False, {'characters in duplicate word 10grams': dup10gram_char / total_character_count}

    # rule 10 & 11: The fraction of duplicate sentences and characters in duplicate sentences must be < 0.30 and < 0.20
    # duplicated_sents_ratio, duplicated_chars = count_duplicate_sentences(line)
    # if duplicated_sents_ratio > 0.30:
    #     return False, {'duplicated sentences ratio:', duplicated_sents_ratio}
    # if duplicated_chars / total_character_count > 0.20:
    #     return False, {'duplicated sentences\' chars ratio:', duplicated_chars / total_character_count}

    return True, None


def count_ngram_duplicates(text, n):
    # Count duplicate ngrams
    ngram_counts = {}
    # Count n-grams and their occurences
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    # Sort n-grams in descending order of occurrences
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
    # For n [2,4], we calculate the proportion of characters contained in the n-gram with the highest number of occurrences
    if sorted_ngrams and sorted_ngrams[0]:
        if n <= 4:
            # Only tackle with n-gram with the highest number of occurrences
            ngram = sorted_ngrams[0][0]
            ngram_counts[ngram] -= 1  # Avoid counting the number of repetitions with the n-gram itself
            return ngram_counts[ngram] * len(ngram)
        # For n in [5,10], we compute the proportion of characters contained in all duplicate n-grams
        else:
            duplicate_ngrams = set()  # Store all duplicate n-grams
            overlapping_characters = set()  # Store characters in duplicate n-grams
            for ngram, count in sorted_ngrams:
                if count > 1:
                    duplicate_ngrams.add(ngram)
                    overlapping_characters.update(set(ngram))
            return len(overlapping_characters)
    return 0