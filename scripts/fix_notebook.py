import json
import os

nb_path = 'src/notebooks/analysis.ipynb'

if os.path.exists(nb_path):
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            for line in source:
                if 'total = len(common_ids)' in line:
                    new_source.append(line.replace('total = len(common_ids)', 'total = len(sub1)'))
                    changed = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if changed:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed {nb_path}")
    else:
        print(f"No changes needed for {nb_path}")
else:
    print(f"Notebook not found at {nb_path}")
