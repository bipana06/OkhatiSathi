from drug_named_entity_recognition import find_drugs

def extract_medicine_name(text):
    data = find_drugs(text.split(" "))
    if data:
        drug_name = data[0][0]["name"]
        matching_name = data[0][0]["matching_string"]
        synonyms = data[0][0]["synonyms"]
        return drug_name, matching_name, synonyms
    else:
        return None, None, None