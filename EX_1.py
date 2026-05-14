import csv

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        header = data[0]
        examples = data[1:]
    return header, examples

def candidate_elimination(examples):
    num_attributes = len(examples[0]) - 1
    

    S = ['0'] * num_attributes
    for ex in examples:
        if ex[-1].lower() == "yes":
            S = ex[:-1]
            break
    
  
    G = [['?' for _ in range(num_attributes)]]

    for i, example in enumerate(examples):
        attributes, label = example[:-1], example[-1].lower()

        if label.lower() == "yes":  
            for j in range(num_attributes):
                if S[j] != attributes[j]:
                    S[j] = '?'
            

            G = [g for g in G if all(g[k] == '?' or g[k] == attributes[k] for k in range(num_attributes))]

        else: 
            new_G = []
            for g in G:
              
                covers_negative = all(g[k] == '?' or g[k] == attributes[k] for k in range(num_attributes))

                if covers_negative:

                    for k in range(num_attributes):
                        if g[k] == '?':
                            if S[k] != '?' and S[k] != attributes[k]:
                                new_hypothesis = g.copy()
                                new_hypothesis[k] = S[k]

                                if all(new_hypothesis[j] == '?' or new_hypothesis[j] == S[j] for j in range(num_attributes)):
                                    if new_hypothesis not in new_G:
                                        new_G.append(new_hypothesis)
                else:
               
                    new_G.append(g)
            
            G = new_G

    return S, G

if __name__ == "__main__":
    try:
        header, examples = load_data("training_data_set_for_ex_1.csv")
        S, G = candidate_elimination(examples)
        
        print("\n--- Final Result ---")
        print("Final Specific Boundary (S):", S)
        print("Final General Boundary (G):", G)
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found. Please create the file first.")
