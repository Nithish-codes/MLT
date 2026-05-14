import csv
import math

def load_data(filename):
 
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data

def calculate_entropy(data):

    if not data:
        return 0
    

    total_rows = len(data)
    target_counts = {}
    for row in data:
        label = row['PlayTennis']
        target_counts[label] = target_counts.get(label, 0) + 1

    entropy = 0
    for count in target_counts.values():
        probability = count / total_rows
        entropy -= probability * math.log2(probability)
        
    return entropy

def calculate_information_gain(data, attribute):

    total_entropy = calculate_entropy(data)
    total_rows = len(data)

    attribute_values = {}
    for row in data:
        val = row[attribute]
        if val not in attribute_values:
            attribute_values[val] = []
        attribute_values[val].append(row)
    

    weighted_entropy = 0
    for val, subset in attribute_values.items():
        probability = len(subset) / total_rows
        weighted_entropy += probability * calculate_entropy(subset)

    return total_entropy - weighted_entropy

def id3(data, attributes):

    labels = [row['PlayTennis'] for row in data]
    if len(set(labels)) == 1:
        return labels[0]
    

    if not attributes:

        return max(set(labels), key=labels.count)
    
 
    best_attr = None
    max_gain = -1
    
    for attr in attributes:
        gain = calculate_information_gain(data, attr)
        if gain > max_gain:
            max_gain = gain
            best_attr = attr
            

    tree = {best_attr: {}}
    

    remaining_attributes = [a for a in attributes if a != best_attr]
    

    unique_values = set(row[best_attr] for row in data)
    for val in unique_values:
        subset = [row for row in data if row[best_attr] == val]
        subtree = id3(subset, remaining_attributes)
        tree[best_attr][val] = subtree
        
    return tree

def classify(sample, tree):

    if not isinstance(tree, dict):
        return tree

    root_attr = list(tree.keys())[0]

    sample_val = sample.get(root_attr)

    if sample_val in tree[root_attr]:
        subtree = tree[root_attr][sample_val]
        return classify(sample, subtree)
    else:
        return "Unknown" 

if __name__ == "__main__":
    filename = "training_data_set_for_ex_2.csv"
    
    try:
        data = load_data(filename)
        

        attributes = list(data[0].keys())
        attributes.remove('PlayTennis')
        
        print("Building Decision Tree...")
        decision_tree = id3(data, attributes)
        

        import pprint
        print("\nGenerated Decision Tree:")
        pprint.pprint(decision_tree)
        
        print("\n-------------------------")

        new_sample = {
            'Outlook': 'Rain',
            'Temperature': 'Cool',
            'Humidity': 'High',
            'Wind': 'Strong'
        }
        
        print(f"Classifying new sample: {new_sample}")
        prediction = classify(new_sample, decision_tree)
        print(f"Prediction: {prediction}")

    except FileNotFoundError:
        print(f"Error: {filename} not found. Please create the CSV file.")
