import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import umap.umap_ as umap

data = np.loadtxt('new_test.txt', delimiter=',')
print("Data shape:", data.shape)
# Normalize the data
# data_normalized = data / 255.0
# pca = PCA()
# pca.fit(data_normalized)
# data_pca = pca.fit_transform(data_normalized)
# print("Shape of PCA-transformed data:", data_pca.shape)

# Visualize the first image
num_images = 9  # Number of images to display
plt.figure(figsize=(5, 5))

for i in range(num_images):
    # Get the image vector and reshape
    image_vector = data[i]
    image_matrix = image_vector.reshape(28, 28)
    # Add a subplot
    plt.subplot(3, 3, i+1)
    plt.imshow(image_matrix, cmap='gray')
    plt.title(f'Image {i+1}')
    plt.axis('off')

#plt.tight_layout()
plt.show()

class KMeans:
    def __init__(self, n_clusters, max_iter=10000, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        # Initialize centroids randomly from the data points
        self.centroids = data[np.random.choice(range(len(data)), self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Assign clusters
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            # Compute new centroids
            new_centroids = np.array([data[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
            self.centroids = new_centroids

        return self

    # predict the labels
    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    # calculate sum of squared distance error
    def sse(self, data):
        # Calculate distances from each point to its assigned centroid
        distances = np.sqrt(((data - self.centroids[self.labels, :])**2).sum(axis=1))
        # Sum of squared distances
        return np.sum(distances**2)

scaler = MinMaxScaler()
images_normalized = scaler.fit_transform(data)

# Dimension reduction 
pca = PCA(n_components=50, random_state=42)
data_pca = pca.fit_transform(images_normalized)

umap_reducer = umap.UMAP(
    n_components=50,
    n_neighbors=15,
    min_dist=0.0,
    metric='euclidean',
    random_state=42
)
embedding = umap_reducer.fit_transform(images_normalized)

# We know that K value is 10
kmeans = KMeans(n_clusters=10)
kmeans.fit(embedding)
predicted_labels = kmeans.predict(embedding)

sse_value = kmeans.sse(embedding)
print(f'Sum of Squared Errors (SSE): {sse_value:.2f}')

file_name = "format_gridSearch.txt"
with open(file_name, 'w+') as file:
    for prediction in predicted_labels:
        file.write(str(prediction) + '\n')

print("+++++++++ Prediction written to file: {} +++++++++".format(file_name))