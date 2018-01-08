import math
import random
import functools
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt
# set matplotlib to display all plots inline with the notebook

max_images = 60000
labels = []
images = []
debug = True

def display_digit(array, label = ""):    
    plt.figure()
    fig = plt.imshow(array.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if label != "":
        plt.title("Given label: " + str(label))
    else:
        plt.title("No label.")
    plt.show()

def read_labels():
    with open("/mnt/d/Projects/MNIST Test/train-labels-idx1-ubyte/train-labels.idx1-ubyte", "rb") as f:
        data = f.read()
        print("Magic number: "+str(int.from_bytes(data[0:4], byteorder='big')))
        print("Number of labels: "+str(int.from_bytes(data[4:8], byteorder='big')))
        
        file_byte_index = 8
        label_counter = 0

        while file_byte_index<len(data):    
            byte = data[file_byte_index]
            file_byte_index +=1
            labels.append(byte)
            label_counter +=1
            if debug:   
                if(label_counter%100==0):
                    print(label_counter)
                if label_counter > max_images:
                    break

def read_images():
    with open("/mnt/d/Projects/MNIST Test/train-images-idx3-ubyte/train-images.idx3-ubyte", "rb") as f:
        data =  f.read()
        print("Magic number: "+str(int.from_bytes(data[0:4], byteorder='big')))
        print("Number of images: "+str(int.from_bytes(data[4:8], byteorder='big')))
        print("Number of rows: "+str(int.from_bytes(data[8:12], byteorder='big')))
        print("Number of cols: "+str(int.from_bytes(data[12:16], byteorder='big')))

        file_byte_index = 16
        image_counter = 0
        while file_byte_index<len(data):
            
            image = []
            for i in range(28*28):
                byte = data[file_byte_index]
                file_byte_index += 1
                image.append(byte)
            images.append(np.array(image))
            image_counter+=1
            
            if debug:    
                if(image_counter%100==0):
                    print(image_counter)
                if image_counter > max_images:
                    break

def init_centroids(images, k):
    """
    randomly pick some k centers from the data as starting values
    for centroids. Remove labels.
    """
    return random.sample(images, k)

def sum_cluster(cluster):
    """
    from http://stackoverflow.com/a/20642156
    element-wise sums a list of arrays.
    """
    # assumes len(cluster) > 0
    sum_ = functools.reduce((lambda x, y: x+y), cluster)
    return sum_

def mean_cluster(cluster):
    """
    compute the mean (i.e. centroid at the middle)
    of a list of vectors (a cluster):
    take the sum and then divide by the size of the cluster.
    """
    sum_of_points = sum_cluster(cluster)
    mean_of_points = sum_of_points * (1.0 / len(cluster))
    return mean_of_points

def form_clusters(images, centroids):
    """
    Create an empty list for each centroid. The list will contain all the data closest to that centroid
    """
    centroids_indices = range(len(centroids))
    clusters = {c: [] for c in centroids_indices}

    for image in images:
        smallest_distance = math.inf
        for centroid_index in centroids_indices:
            centroid = centroids[centroid_index]
            distance = np.linalg.norm(image - centroid)
            if distance < smallest_distance:
                closest_centroid_index = centroid_index
                smallest_distance = distance
        clusters[closest_centroid_index].append(image)
    return clusters.values()

def calculate_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids

def repeat_until_convergence(images, initial_clusters, initial_centroids):
    iteration = 0
    previous_max_difference = 0

    
    while True:
        # centroids from beginning of iteration
        old_centroids = initial_centroids
        #calculate new centroids
        initial_centroids = calculate_centroids(initial_clusters)
        #form new clusters
        initial_clusters = form_clusters(images, initial_centroids)

        #
        differences = map(lambda a,b: np.linalg.norm(a-b), initial_centroids, old_centroids)
        max_difference = max(differences)

        if debug:
            print("iteration: "+str(iteration))
            iteration+=1
            print("max difference: " +str(max_difference))
            print("pmax difference: " +str(previous_max_difference))

        if max_difference == 0:
            break
        
        difference_change = abs((max_difference-previous_max_difference)/np.mean([max_difference, previous_max_difference])) * 100
        previous_max_difference = max_difference

    return initial_clusters, initial_centroids



def cluster(images, k):
    initial_centroids = init_centroids(images, k)
    initial_clusters = form_clusters(images, initial_centroids)
    final_clusters, final_centroids = repeat_until_convergence(images, initial_clusters, initial_centroids)
    return final_clusters, final_centroids

    

read_labels()
print("done reading labels.")
read_images()
print("done reading images.")

clusters, centroids = cluster(images, 9)
for centroid in centroids:
    display_digit(centroid)











