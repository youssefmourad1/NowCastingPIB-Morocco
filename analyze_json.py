import json
from datetime import datetime

file_path = '/Users/Apple/Desktop/projects/Tennis AI v2.0/Tennis AI Analysis/extract.json'

with open(file_path, 'r') as f:
    data = json.load(f)

items = data.get('data', [])
print(f"Total items: {len(items)}")

if items:
    # Print first few items to see structure again
    for item in items[:5]:
        attrs = item.get('attributes', {})
        print(f"Date: {attrs.get('field_seance_date')}, Value: {attrs.get('field_index_value')}")

    # Check for any other fields that might indicate the index name
    print("\nKeys in attributes of first item:")
    print(items[0].get('attributes', {}).keys())
    
    # Check if there are different types
    types = set(item.get('type') for item in items)
    print(f"\nTypes found: {types}")
