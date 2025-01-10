import fitz
import os
import stanza
import json

def extract_between_expressions(pdf_path, expr1, expr2):
    doc = fitz.open(pdf_path)
    indices_expr1 = []
    indices_expr2 = []

    # Convert expressions to lowercase for case insensitive search
    expr1_lower = expr1.lower()
    expr2_lower = expr2.lower()

    # Search for expressions in the document
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Convert text to lowercase
        text_lower = text.lower()

        # Find all occurrences of expr1
        start = 0
        while True:
            start = text_lower.find(expr1_lower, start)
            if start == -1:
                break
            indices_expr1.append((page_num, start))
            start += len(expr1_lower)  # Move past the last found index

        # Find all occurrences of expr2
        start = 0
        while True:
            start = text_lower.find(expr2_lower, start)
            if start == -1:
                break
            indices_expr2.append((page_num, start))
            start += len(expr2_lower)  # Move past the last found index

    # Check if we have at least two occurrences of each expression
    if len(indices_expr1) < 2 or len(indices_expr2) < 2:
        print("Not enough occurrences found.")
        return

    # Get the indices of the second occurrences
    second_expr1_index = indices_expr1[1]
    second_expr2_index = indices_expr2[1]

    # Extract content between the two indices
    start_page, start_pos = second_expr1_index
    end_page, end_pos = second_expr2_index

    content = []
    
    # If both expressions are on the same page
    if start_page == end_page:
        page = doc[start_page]
        text = page.get_text()
        content.append(text[start_pos:end_pos])
    else:
        # Handle content across pages
        # Extract content from start page
        page = doc[start_page]
        text = page.get_text()
        content.append(text[start_pos:])

        # Extract full content of pages in between
        for p in range(start_page + 1, end_page):
            content.append(doc[p].get_text())

        # Extract content from end page
        page = doc[end_page]
        text = page.get_text()
        content.append(text[:end_pos])

    # Join and return the extracted content
    return ''.join(content)

def extract_from_all_pdfs(folder_path, expr1, expr2):
    """Extract content from all PDF files in the specified folder."""
    success = 0
    corpus = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            result = extract_between_expressions(pdf_path, expr1, expr2)
            if result:
                # Save or print the result as needed
                success += 1
                #print(f"Extracted content from {filename}:\n{result}\n")
                corpus.append(result)
                print("sucsess:", success)
            else:
                print(f"No valid content extracted from {filename}.")
    return corpus


# Usage
folder_path = "/Users/songdelin/Desktop/fyp/annual_reports"
expression1 = "item 1a."
expression2 = "item 1b."
corpus = extract_from_all_pdfs(folder_path, expression1, expression2)
corpus_length = len(corpus)
corpus_segmentation = []
num = 0
nlp = stanza.Pipeline(lang='en', processors='tokenize')
for text in corpus:
    
    
    doc = nlp(text)
    list = []
    
    
    for sentence in doc.sentences:
        
        new_sentence = sentence.text.replace("\n", " ")
        
        new_sentence = new_sentence.replace("\u2019", "'")
        list.append(new_sentence)
    
    corpus_segmentation.append(list)
    num += 1
    print(f'complete: {num}/{corpus_length}')

with open('corpus_item1a.json', 'w') as f:
    json.dump(corpus_segmentation, f)









    

