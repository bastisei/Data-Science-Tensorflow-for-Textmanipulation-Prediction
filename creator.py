import random
import string
import pandas as pd
import csv

def random_string(length=10):
    letters = string.ascii_letters + string.digits + '#;/,.'
    return ''.join(random.choice(letters) for i in range(length))

def combine(s1, s2):
    return f"{s1}{s2}", 'Combine'

def replace(input_string):
    if input_string and len(input_string) > 1:
        old_char = random.choice([char for char in input_string if char.isalnum() or char in '#;/,.'])
        new_char = random.choice(string.ascii_letters + string.digits + '#;/,.')
        result = input_string.replace(old_char, new_char, 1)
        if result != input_string:
            return result, 'Replace'
    return input_string, 'Replace (no action)'

def string_length(input_string):
    if input_string:
        return len(input_string), 'Length'
    return 0, 'Length (no action)'

def extract(input_string, start, length):
    if input_string and 0 <= start < len(input_string) and start + length <= len(input_string):
        return input_string[start:start+length], 'Extract'
    return input_string, 'Extract (no action)'

def first_characters(input_string, n):
    if input_string and 0 < n <= len(input_string):
        return input_string[:n], 'First Characters'
    return input_string, 'First Characters (no action)'

def last_characters(input_string, n):
    if input_string and 0 < n <= len(input_string):
        return input_string[-n:], 'Last Characters'
    return input_string, 'Last Characters (no action)'

def range_characters(input_string, start, end):
    if end > start and end <= len(input_string):
        return input_string[start:end], 'Range'
    return input_string, 'Range (no action)'

def text_before_delimiter(input_string, delimiter):
    parts = input_string.split(delimiter)
    if len(parts) > 1:
        return parts[0], 'Text before Delimiter'
    return input_string, 'Text before Delimiter (no action)'

def text_after_delimiter(input_string, delimiter):
    parts = input_string.split(delimiter)
    if len(parts) > 1:
        return parts[1], 'Text after Delimiter'
    return input_string, 'Text after Delimiter (no action)'

def text_between_delimiters(input_string, start_delim, end_delim):
    start = input_string.find(start_delim)
    if start != -1:
        start += len(start_delim)
        end = input_string.find(end_delim, start)
        if end != -1:
            return input_string[start:end], 'Text between Delimiters'
    return input_string, 'Text between Delimiters (no action)'

def remove_characters(input_string, chars_to_remove):
    result = "".join(char for char in input_string if char not in chars_to_remove)
    if result != input_string and result:
        return result, 'Remove Characters'
    return input_string, 'Remove Characters (no action)'

def keep_characters(input_string, chars_to_keep):
    result = "".join(char for char in input_string if char in chars_to_keep)
    if result:
        return result, 'Keep Characters'
    return input_string, 'Keep Characters (no action)'

def generate_data(num_samples=200000):
    data = []
    transformations = [
        'Combine', 'Replace', 'Length', 'Extract', 'First Characters', 'Last Characters',
        'Text before Delimiter', 'Text after Delimiter', 'Text between Delimiters',
        'Remove Characters', 'Keep Characters'
    ]
    delimiters = [',', ';', '/']
    for _ in range(num_samples):
        original = random_string(random.randint(5, 15))
        transformation_type = random.choice(transformations)
        result, trans_description = '', ''

        if transformation_type == 'Combine':
            result, trans_description = combine(original, random_string(random.randint(1, 10)))
        elif transformation_type == 'Replace':
            result, trans_description = replace(original)
        elif transformation_type == 'Length':
            result, trans_description = string_length(original)
        elif transformation_type in ['First Characters', 'Last Characters', 'Extract']:
            n = random.randint(1, 10)
            if transformation_type == 'Extract':
                start = random.randint(0, len(original) - 1)
                result, trans_description = extract(original, start, n)
            elif transformation_type == 'First Characters':
                result, trans_description = first_characters(original, n)
            elif transformation_type == 'Last Characters':
                result, trans_description = last_characters(original, n)
        elif transformation_type in ['Text before Delimiter', 'Text after Delimiter', 'Text between Delimiters']:
            delimiter = random.choice(delimiters)
            if transformation_type == 'Text before Delimiter':
                result, trans_description = text_before_delimiter(original, delimiter)
            elif transformation_type == 'Text after Delimiter':
                result, trans_description = text_after_delimiter(original, delimiter)
            else:
                end_delim = random.choice(delimiters)
                result, trans_description = text_between_delimiters(original, delimiter, end_delim)
        elif transformation_type == 'Remove Characters':
            chars_to_remove = random_string(random.randint(1, 5))
            result, trans_description = remove_characters(original, chars_to_remove)
        elif transformation_type == 'Keep Characters':
            chars_to_keep = random_string(random.randint(1, 5))
            result, trans_description = keep_characters(original, chars_to_keep)

        if 'no action' not in trans_description:
            data.append((original, str(result), trans_description))

    return data

def save_data_to_csv(data, filename='text_transformation_examples.csv'):
    df = pd.DataFrame(data, columns=['Input', 'Output', 'Transformation'])
    df.to_csv(
        filename, 
        index=False, 
        sep='|', 
        quoting=csv.QUOTE_NONE, 
        escapechar='\\'
    )

data_examples = generate_data()
save_data_to_csv(data_examples)
