import json
from openai import OpenAI
import pandas as pd

client = OpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1"
)

#Extraction template
template = {
    # 1. MMAD
    "number_of_participants": "string",
    "protest_issue": "string",
    "government_versus_non_government_side": ["anti_gov", "pro_gov", "non_gov_target"],
    "scope_of_protest": ["national", "regional", "local", "NA"],
    "violence_by_participants": ["NA", "no_violence", "property_damage_or_clashes", "injuries", "deaths"],
    "security_forces_engagement": ["NA", "no_presence", "presence_only", "physical_intervention", "lethal_intervention"],

    # 2. MMD
    "violence_by_protesters": ["no", "yes"],
    "identity_of_protesters": "string",
    "number_of_participants": "string",  
    "number_participants": "integer",  
    "number_participants_category": ["50-99", "100-999", "1000-1999", "2000-4999", "5000-10000", ">10000"],
    "protester_demand_1": ["labor_wage_dispute","land_farm_issue","police_brutality","political_behavior_process","price_tax_policy","remove_corrupt_person","social_restriction"],
    "protesterdemand_2": ["labor_wage_dispute","land_farm_issue","police_brutality","political_behavior_process","price_tax_policy","remove_corrupt_person","social_restriction"],
    "response_by_government_1": ["accommodation","arrests","beatings","crowd_dispersal","ignore","killings","shootings"],
    "response_by_government_2": ["accommodation","arrests","beatings","crowd_dispersal","ignore","killings","shootings"],

    # 3. NAVCO
    "goals_of_protesters": ["regime_change","institutional_reform","policy_change","secession","autonomy","anti_occupation","unknown"],
    "protester_tactic_choice": ["violent","nonviolent","mixed"],
    "protester_choice_non_violent_category": ["persuasion","protest","noncooperation","intervention","political_engagement"],
    "non_violence_commission": ["omission","commission","ambiguous"],
    "non_violence_concentration": ["dispersion","concentration","mixed"],
    "non_violence_tactic_description": "string",  
    "how_government_responded": ["full_accommodation","material_concessions","nonmaterial_concessions","neutral","nonmaterial_non-physical_repression","material_physical_repression","lethal_repression"],
    "number_fatal_casualties": "integer",  
    "number_injuries": "integer",  
    "number_of_event_participants": "string",  
    "damage_level": ["minor","significant","substantial"],
    "level_economic_impact": ["little_or_none","significant_local_regional","heavy_national"],
    "economic_impact_details": "string",

    # 4. SCAD
    "event_type": ["org_demo","spont_demo","org_riot","spont_riot","general_strike","limited_strike","pro_gov_violence","anti_gov_violence","extra_gov_violence","intra_gov_violence","NA"],
    "escalation": ["no_escalation","org_demo","spont_demo","org_riot","spont_riot","general_strike","limited_strike","pro_gov_violence","anti_gov_violence","extra_gov_violence","intra_gov_violence"],
    "central_government_target": ["no","yes"],
    "regional_government_target": ["no","yes"],
    "number_participants_category": ["less_than_10","10_to_100","101_to_1000","1001_to_10000","10001_to_100000","100001_to_1000000","over_1000000","unknown"],
    "number_deaths": "integer",  
    "government_repression": ["none","non_lethal","lethal"],
    "protest_issue_category_1": ["elections","economy_jobs","food_water_subsistence","environmental_degradation","ethnic_discrimination","religious_discrimination","education","foreign_affairs","domestic_war_terrorism","human_rights_democracy","pro_government","economic_resources","other","unknown"],
    "protest_issue_category_2": ["elections","economy_jobs","food_water_subsistence","environmental_degradation","ethnic_discrimination","religious_discrimination","education","foreign_affairs","domestic_war_terrorism","human_rights_democracy","pro_government","economic_resources","other"],
    "gender_lgbtq_related": ["female_event", "lgbtq_issue"]
}

# Texts
df_labels = pd.read_csv("labels.csv")

# Drop missing values and convert notes column into list of documents
documents = df_labels["NOTES"].dropna().astype(str).tolist()

results = []

# Loop extraction on each document
for doc in documents:
    chat_response = client.chat.completions.create(
        model="numind/NuExtract-2.0-8B",  
        temperature=0,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": doc}],
        }],
        extra_body={
            "chat_template_kwargs": {
                "template": json.dumps(template)
            }
        }
    )
    
    extracted = chat_response.choices[0].message.content
    
    print("Input Document:")
    print(doc)
    print("Extracted Info:")
    print(extracted)
    print("=" * 80)
    
    results.append({
        "document": doc,
        "extracted_info": extracted
    })

# Save 
df = pd.DataFrame(results)
df.to_csv("extracted_event_data.csv", index=False)
print("Results saved to extracted_event_data.csv")
