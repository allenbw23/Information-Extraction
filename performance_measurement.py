#Import packages
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

#Import data
labels = pd.read_csv(".../labels.csv")
extracted = pd.read_csv(".../extracted_event_data.csv")

#Expand extracted JSON into columns
extracted_expanded = extracted["extracted_info"].apply(json.loads).apply(pd.Series)
extracted = pd.concat([extracted["document"], extracted_expanded], axis=1)

#Make common index
labels["document"] = labels["NOTES"]

#Drop columns that dont help
labels = labels.drop(columns=["ACTOR1","ACTOR2","COUNTRY","EVENT_DATE","EVENT_ID_CNTY","EVENT_TYPE","ISO","LOCATION","NOTES","SUB_EVENT_TYPE","YEAR","annotation_id","annotator","created_at","id","lead_time", "updated_at"])

#Make map dictionary
mapping = {
    "max_partviolence": "violence_by_participants", #categorical
    "max_secengagement": "security_forces_engagement", # categorical
    "mmad_issue": "protest_issue", # string
    "mmad_max_scope": "scope_of_protest", #categorical
    "mmad_mean_avg_numparticipants": "number_of_participants", # string
    "mmad_side": "government_versus_non_government_side", #categorical
    "mmd_participants": "identity_of_protesters", #string
    "mmd_participants_category": "number_participants_category", # categorical
    "mmd_protesterdemand_1": "protester_demand_1", #categorical
    "mmd_protesterdemand_2": "protesterdemand_2", #categorical
    "mmd_protesteridentity": "identity_of_protesters", #string
    "mmd_protesterviolence": "violence_by_protesters", # categorical
    "mmd_protestnumber": "number_participants", #string
    "mmd_stateresponse_1": "response_by_government_1", #categorical
    "mmd_stateresponse_2": "response_by_government_2", #categorical
    "navco_camp_goals": "goals_of_protesters", #categorical 
    "navco_damage": "damage_level", #categorical
    "navco_fatal_casu": "number_fatal_casualties", #numeric integer
    "navco_injuries": "number_injuries", #numeric integer
    "navco_num_partic_event": "number_of_event_participants", #string
    "navco_nv_categ": "protester_choice_non_violent_category", #categorical
    "navco_nv_commission": "non_violence_commission", #categorical
    "navco_nv_concentration": "non_violence_concentration", #categorical
    "navco_nv_tactic_ns": "non_violence_tactic_description", #string
    "navco_st_posture": "how_government_responded", # categorical
    "navco_tactic_choice": "protester_tactic_choice", #categorical
    "navco_v_tactic_ns": "non_violence_tactic_description", #string
    "scad_cgovtarget": "central_government_target", #categorical
    "scad_escalation": "escalation", #categorical
    "scad_etype": "event_type", #categorical
    "scad_gender_tags": "gender_lgbtq_related", #categorical
    "scad_issues_1": "protest_issue_category_1", #categorical
    "scad_issues_2": "protest_issue_category_2", #categorical
    "scad_ndeath": "number_deaths", #numerical integer
    "scad_npart": "number_participants_category", #categorical
    "scad_repress": "government_repression", # categorical
    "scad_rgovtarget": "regional_government_target" #categorical
}

#Rename columns to approximately match
labels = labels.rename(columns={label_col: f"{label_col}_label" for label_col in mapping.keys()})
extracted = extracted.rename(columns={extract_col: f"{label_col}_extract" 
                                      for label_col, extract_col in mapping.items()})

#Combine the dataframes based on the shared documents column
combined = pd.merge(labels, extracted, left_on="document", right_on="document", suffixes=("_label", "_extract"))

#Start with categorical performance assessment

#identify categorical columns
categorical_vars = [
    "max_partviolence",
    "max_secengagement",
    "mmad_max_scope",
    "mmad_side",
    "mmd_participants_category",
    "mmd_protesterdemand_1",
    "mmd_protesterdemand_2",
    "mmd_protesteridentity",
    "mmd_protesterviolence",
    "mmd_stateresponse_1",
    "mmd_stateresponse_2",
    "navco_camp_goals",
    "navco_damage",
    "navco_nv_categ",
    "navco_nv_commission",
    "navco_nv_concentration",
    "navco_st_posture",
    "navco_tactic_choice",
    "scad_cgovtarget",
    "scad_escalation",
    "scad_etype",
    "scad_gender_tags",
    "scad_issues_1",
    "scad_issues_2",
    "scad_npart",
    "scad_repress",
    "scad_rgovtarget"
]

#Split by column and start loop to get accuracy
categorical_label_cols = [f"{var}_label" for var in categorical_vars if f"{var}_label" in combined.columns]
categorical_extract_cols = [f"{var}_extract" for var in categorical_vars if f"{var}_extract" in combined.columns]
results = {}
for var in categorical_vars:
    label_col = f"{var}_label"
    extract_col = f"{var}_extract"
    
    if label_col in combined.columns and extract_col in combined.columns:
        acc = (combined[label_col] == combined[extract_col]).mean()
        results[var] = round(acc, 3)

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
print(results_df)
print("Overall Accuracy:", results_df["Accuracy"].mean())

#Numeric performance assessment

#Identify numeric columns
numerical_vars = [
    "navco_fatal_casu",   
    "navco_injuries",     
    "scad_ndeath"        
]

#Define function to extract number from variable
import ast

def extract_number(x):
    try:
        if isinstance(x, str):
            # Try to parse if it looks like JSON/dict
            if x.startswith("[") or x.startswith("{"):
                parsed = ast.literal_eval(x)  # safely evaluate
                # Handle structures like [{"number": 4}]
                if isinstance(parsed, list) and "number" in parsed[0]:
                    return parsed[0]["number"]
            return float(x)  # try convert directly
        return float(x) if pd.notnull(x) else None
    except:
        return None

# Apply to numerical columns in combined
for var in ["navco_fatal_casu", "navco_injuries", "scad_ndeath"]:
    for suffix in ["_label", "_extract"]:
        col = f"{var}{suffix}"
        if col in combined.columns:
            combined[col] = combined[col].apply(extract_number)

#Run loop and print results
results_num = {}

for var in ["navco_fatal_casu", "navco_injuries", "scad_ndeath"]:
    label_col = f"{var}_label"
    extract_col = f"{var}_extract"
    
    if label_col in combined.columns and extract_col in combined.columns:
        # Drop rows with NaN in either column
        temp = combined[[label_col, extract_col]].dropna()
        y_true = temp[label_col]
        y_pred = temp[extract_col]
        
        if len(temp) > 0:  # only compute if data exists
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            corr = y_true.corr(y_pred)
            
            results_num[var] = {
                "n_obs": len(temp),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "Corr": round(corr, 3)
            }


results_num_df = pd.DataFrame(results_num).T
print(results_num_df)

#Free text performance assessment

#Identify free text columns
string_vars = [
    "mmad_issue",                
    "mmd_participants",           
    "mmd_protesteridentity",     
    "navco_nv_tactic_ns",         
    "navco_v_tactic_ns"           
]

#Run loop for EXACT match assessment
results_str = {}

for var in string_vars:
    label_col = f"{var}_label"
    extract_col = f"{var}_extract"
    
    if label_col in combined.columns and extract_col in combined.columns:
        temp = combined[[label_col, extract_col]].dropna() 
        y_true = temp[label_col]
        y_pred = temp[extract_col]
        
        acc = (y_true == y_pred).mean()  
        results_str[var] = {
            "n_obs": len(temp),
            "Exact Match Accuracy": round(acc, 3)
        }

results_str_df = pd.DataFrame(results_str).T
print(results_str_df)

#Show where actual mismatches were
for var in string_vars:
    label_col = f"{var}_label"
    extract_col = f"{var}_extract"
    
    if label_col in combined.columns and extract_col in combined.columns:
        mismatches = combined[combined[label_col] != combined[extract_col]][[label_col, extract_col]]
        print(f"\nMismatches for {var}:")
        print(mismatches.head(10))  

#Run performance assessment on free text with jaccard index
def jaccard_similarity(a, b):
    set_a = set(str(a).lower().split())
    set_b = set(str(b).lower().split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

results_jaccard = {}

for var in string_vars:
    label_col = f"{var}_label"
    extract_col = f"{var}_extract"
    
    if label_col in combined.columns and extract_col in combined.columns:
        temp = combined[[label_col, extract_col]].dropna()
        sims = temp.apply(lambda row: jaccard_similarity(row[label_col], row[extract_col]), axis=1)
        results_jaccard[var] = {
            "n_obs": len(temp),
            "Avg Jaccard": round(sims.mean(), 3)
        }

results_jaccard_df = pd.DataFrame(results_jaccard).T
print(results_jaccard_df)

# Run performance assessment on free text with token matching
def token_match_ratio(a, b):
    tokens_a = str(a).lower().split()
    tokens_b = str(b).lower().split()
    if not tokens_a:
        return 0
    return sum(1 for t in tokens_a if t in tokens_b) / len(tokens_a)

results_token = {}

for var in string_vars:
    label_col = f"{var}_label"
    extract_col = f"{var}_extract"
    
    if label_col in combined.columns and extract_col in combined.columns:
        temp = combined[[label_col, extract_col]].dropna()
        ratios = temp.apply(lambda row: token_match_ratio(row[label_col], row[extract_col]), axis=1)
        results_token[var] = {
            "n_obs": len(temp),
            "Avg Token Match": round(ratios.mean(), 3)
        }

results_token_df = pd.DataFrame(results_token).T
print(results_token_df)
