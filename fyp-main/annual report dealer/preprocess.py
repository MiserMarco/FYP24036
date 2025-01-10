import json
import random
import pandas as pd
import shutil
import os






def get_file_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
    #randomly select 100 sentences
        usable_data = [i for i in range(len(data)) if len(data[i]) > 100]
        
        return {'raw_data': data, 'usable_data': usable_data}

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON.")
        return


def extract_sentences_group(data, usable_data, num_sentence):
    selected_index = []
    
    while(len(selected_index) < num_sentence):
        random_annual_report = random.randint(1, len(usable_data) - 1)
        random_sentence_index = random.randint(10, 80)
        index = [usable_data[random_annual_report], random_sentence_index]
        if index not in selected_index:
            selected_index.append(index)
    
    
    return selected_index

#window_size is the number of sentences in a group, ideally an odd number
def get_sentence_group(data, selected_index, window_size):
    selected_sentences_group = [''.join([data[center[0]][center[1]+i] for i in range(int(-(window_size-1)/2), int((window_size+1)/2))]) for center in selected_index]
    return selected_sentences_group
    

def divide_to_subfiles(sentences, group_size, folder_path):
    group_num = len(sentences) // group_size
    for i in range(group_num):
        sub_sentences = []
        for j in range(group_size):
            sub_sentences.append(sentences[i * group_size + j])
            
        df = pd.DataFrame({'Risky Sentences':sub_sentences, 'T/F':['N/A' for i in range(len(sub_sentences))]})
        csv_file_path = f'batch{i}.csv'
        df.to_csv(csv_file_path, index=False)
        # Ensure the destination folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Move the CSV file to the destination folder
        shutil.move(csv_file_path, os.path.join(folder_path, csv_file_path))

        print(f'File moved to: {os.path.join(folder_path, csv_file_path)}')
            
'''
file_path_item1a = 'corpus_item1a_clean.json'
file_data = get_file_data(file_path_item1a)
usable_data = file_data['usable_data']
raw_data = file_data['raw_data']
selected_index = extract_sentences_group(raw_data, usable_data, 2500)
selected_sentences_group = get_sentence_group(raw_data, selected_index, 5)

folder_path = '/Users/songdelin/Desktop/annual report dealer/item1a'
divide_to_subfiles(selected_sentences_group, 100, folder_path)
'''
'''
df = pd.DataFrame({'Risky Sentences':selected_sentences_group, 'T/F':['N/A' for i in range(len(selected_sentences_group))]})
df.to_csv('item1a_labelled_full.csv', index=False)
'''
'''
file_path_item1 = 'corpus_item1_clean.json'
file_data = get_file_data(file_path_item1)
usable_data = file_data['usable_data']
raw_data = file_data['raw_data']
selected_index = extract_sentences_group(raw_data, usable_data, 2500)
selected_sentences_group = get_sentence_group(raw_data, selected_index, 5)

folder_path = '/Users/songdelin/Desktop/annual report dealer/item1'
divide_to_subfiles(selected_sentences_group, 100, folder_path)
'''
'''
df = pd.DataFrame({'Risky Sentences':selected_sentences_group, 'T/F':['N/A' for i in range(len(selected_sentences_group))]})
df.to_csv('item1_labelled_full.csv', index=False)
'''

file_path_item7 = 'corpus_item7_clean.json'
file_data = get_file_data(file_path_item7)
usable_data = file_data['usable_data']
raw_data = file_data['raw_data']
selected_index = extract_sentences_group(raw_data, usable_data, 2500)
selected_sentences_group = get_sentence_group(raw_data, selected_index, 5)

folder_path = '/Users/songdelin/Desktop/annual report dealer/item7'
divide_to_subfiles(selected_sentences_group, 100, folder_path)
'''
df = pd.DataFrame({'Risky Sentences':selected_sentences_group, 'T/F':['N/A' for i in range(len(selected_sentences_group))]})
df.to_csv('item1_labelled_full.csv', index=False)
'''
