import csv
import math
import random

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return [[float(row['X']), float(row['Y'])] for row in reader]

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i])**2 for i in range(len(p1))))


def k_means(data, k, iterations=10):

    centroids = random.sample(data, k)
    
    for _ in range(iterations):
        clusters = [[] for _ in range(k)]

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
        

        for i in range(k):
            if clusters[i]:
                centroids[i] = [sum(p[j] for p in clusters[i])/len(clusters[i]) for j in range(2)]
    return clusters, centroids


def em_algorithm(data, k, iterations=10):

    means = random.sample(data, k)
    weights = [1/k] * k
    
    for _ in range(iterations):

        responsibilities = []
        for point in data:
            probs = []
            for i in range(k):

                dist = euclidean_distance(point, means[i])
                prob = weights[i] * math.exp(-0.5 * dist**2)
                probs.append(prob)
            total_prob = sum(probs)
            responsibilities.append([p/total_prob for p in probs])
            

        for i in range(k):
            weighted_sum_x = 0
            weighted_sum_y = 0
            total_resp = 0
            for j, point in enumerate(data):
                weighted_sum_x += responsibilities[j][i] * point[0]
                weighted_sum_y += responsibilities[j][i] * point[1]
                total_resp += responsibilities[j][i]
            
            means[i] = [weighted_sum_x/total_resp, weighted_sum_y/total_resp]
            weights[i] = total_resp / len(data)
            
    return means, weights


data = load_data('training_data_set_for_ex_7.csv')

print("Running K-Means...")
km_clusters, km_centroids = k_means(data, k=2)
print(f"K-Means Centroids: {km_centroids}")

print("\nRunning EM Algorithm...")
em_means, em_weights = em_algorithm(data, k=2)
print(f"EM Means: {em_means}")
