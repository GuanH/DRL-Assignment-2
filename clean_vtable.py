import pickle
from collections import defaultdict

with open('v_table_c.pkl', 'rb') as file:
    v_table = pickle.load(file)

for table in v_table:
    keys = []
    for k, v in table.items():
        if v < 10:
            keys.append(k)
    print(f'Num : {len(keys)}')
    for k in keys:
        table.pop(k)


with open('v_table_clean.pkl', 'wb') as file:
    pickle.dump(v_table, file)
