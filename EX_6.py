import csv

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def diagnose(data, symptoms):
    
    total = len(data)
    corona_pos = [row for row in data if row['Corona'] == '1']
    corona_neg = [row for row in data if row['Corona'] == '0']
    
    p_corona = len(corona_pos) / total
    p_no_corona = len(corona_neg) / total
    
    
    prob_pos = p_corona
    prob_neg = p_no_corona
    
    for symptom, value in symptoms.items():
       
        pos_match = len([r for r in corona_pos if r[symptom] == str(value)])
        neg_match = len([r for r in corona_neg if r[symptom] == str(value)])
        
        
        prob_pos *= (pos_match + 1) / (len(corona_pos) + 2)
        prob_neg *= (neg_match + 1) / (len(corona_neg) + 2)
        
    
    total_prob = prob_pos + prob_neg
    final_score = prob_pos / total_prob
    
    return final_score


dataset = load_data('training_data_set_for_ex_6.csv')


patient_symptoms = {'Fever': 1, 'Cough': 0, 'BreathBreath': 1}

probability = diagnose(dataset, patient_symptoms)

print(f"Based on symptoms {patient_symptoms}:")
print(f"Probability of Corona: {probability:.2%}")
