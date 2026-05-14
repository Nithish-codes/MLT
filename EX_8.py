import csv
import math
from collections import Counter

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            
            features = [
                float(row['sepal_length']), 
                float(row['sepal_width']), 
                float(row['petal_length']), 
                float(row['petal_width'])
            ]
            dataset.append((features, row['species']))
    return dataset

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def knn_classify(train_set, test_point, k):
    distances = []
    for train_point, label in train_set:
        dist = euclidean_distance(test_point, train_point)
        distances.append((dist, label))
    
    
    distances.sort(key=lambda x: x[0])
    neighbors = [label for dist, label in distances[:k]]
    
    
    return Counter(neighbors).most_common(1)[0][0]


data = load_dataset('training_data_set_for_ex_8.csv')


train_data = data[:4]
test_data = data[4:]
k = 3

print(f"{'Actual':<15} | {'Predicted':<15} | {'Status'}")
print("-" * 45)

correct = 0
for features, actual in test_data:
    predicted = knn_classify(train_data, features, k)
    
    if predicted == actual:
        status = "CORRECT"
        correct += 1
    else:
        status = "WRONG"
    
    print(f"{actual:<15} | {predicted:<15} | {status}")

accuracy = (correct / len(test_data)) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
