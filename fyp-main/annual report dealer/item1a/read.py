import pandas as pd
import chardet
for i in range(25):
    with open(f'batch{i}.csv', 'rb') as f:
        result = chardet.detect(f.read())
        print(result)
    

def readcsv(file):
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())
    
    
    df = pd.read_csv(file, encoding = result['encoding'])
    
    print('sucessfully read', file)

for i in range(25):
    readcsv(f'batch{i}.csv')


