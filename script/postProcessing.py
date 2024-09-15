import re

def extract_entity_from_text(text, entity_name):
    # Example regex pattern to extract weight
    if entity_name == "item_weight":
        match = re.search(r'(\d+\.?\d*)\s*(kg|g|grams)', text, re.IGNORECASE)
        if match:
            return match.group(0)
    # Add more entity extraction logic as needed
    return None
