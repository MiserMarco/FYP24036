import json

# Specify the file path
file_path_item1 = 'corpus_item1.json'

# Read the JSON file
with open(file_path_item1, 'r') as file:
    data = json.load(file)

# Print the data

min_length = float('inf')
shortest_sentence = ''
for report in data:
    for sentence in report:
        if len(sentence) < min_length:
            min_length = len(sentence)
            shortest_sentence = sentence
print("shortest sentence length:", min_length, "\nsentence content:",shortest_sentence)




clean_data = [[sentence for sentence in report if len(sentence) > 25] for report in data]

with open('corpus_item1_clean.json', 'w') as f:
    json.dump(clean_data, f)


