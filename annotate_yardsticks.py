import requests
from urllib.parse import quote_plus
import json

services = "dependency_relations" # "tense_features" # "morphology_features" "new_family"

message = "La commune est la ville ou le village où vous habitez." # ou:ville,village



difficulty_level = "A1"
message_json = json.dumps(message, ensure_ascii=False)
r = requests.post(url="http://192.168.249.77:8080/process_phenomena",data={"raw_text": message_json, # server
                        "difficulty_level": difficulty_level})
# r = requests.post(url="http://0.0.0.0:8080/process_phenomena",data={"raw_text": message_json,      # local
#                         "difficulty_level": difficulty_level})
                        # "services": services})

output_dict = json.loads(r.text)


# yardsticks call
# phenomena_output = json.dumps(output_dict, ensure_ascii=False)
phenomena_output = output_dict

r = requests.post(url="http://192.168.249.77:8166/processing", json={"phenomena": phenomena_output}) # server
# r = requests.post(url="http://0.0.0.0:8166/processing", json={"phenomena": phenomena_output})      # local

print(r.text)
