import requests

import os

data_train = []
with open('train.csv', 'r') as f:
    reader = f.readlines()
    reader = reader[1:]
    for row in reader:
        data_spl = row.split(',')
        data_train.append([data_spl[1], data_spl[2]])

os.makedirs('dataset', exist_ok=True)
for i in data_train[:1000]:
    url = i[1]
    response = requests.get(url)
    file_path = f'dataset/{i[0]}.mp4'
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
    # testfile.retrieve(f'{i[1]}', f'dataset/{i[0]}.mp4')
