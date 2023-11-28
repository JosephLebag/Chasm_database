import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import tkinter as tk
from tkinter import scrolledtext

# Load the dataset from a CSV file
data = pd.read_csv("C:\\Users\\phili\\OneDrive\\Documents\\Dataset_THESIS2.csv")

# Handle missing values (fill missing text with a placeholder)
data['title'].fillna("No Title", inplace=True)
data['description'].fillna("No Description", inplace=True)

# Combine the 'year', 'title', and 'description' into a single column
data['text_data'] = data['year'].astype(str) + " " + data['title'] + " " + data['description']

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Transform the text data into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text_data'])

# Perform Latent Semantic Analysis (LSA) on the TF-IDF matrix
n_components = 2  # Adjust the number of components based on your preference
lsa = TruncatedSVD(n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Determine the optimal number of clusters (e.g., using the Elbow method)
inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1000, n_init=10)
    kmeans.fit(lsa_matrix)
    inertia.append(kmeans.inertia_)

# Perform K-Means clustering with the optimal number of clusters
optimal_k = 2  # Default to the minimum value
for k in range(1, len(inertia) - 1):
    rate_of_change = (inertia[k] - inertia[k + 1]) / (inertia[k - 1] - inertia[k])
    if rate_of_change < 0.5:  # You can adjust this threshold as needed
        optimal_k = k_range[k]
        break

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

kmeans_model = KMeans(n_clusters=optimal_k, random_state=1000, n_init=10)
kmeans_result = kmeans_model.fit_predict(lsa_matrix)

# Your new data point in the same format as the training data
new_text_document = "This is a new document for prediction."

# Transform the new data point into a TF-IDF vector using the same vectorizer
new_data_point_tfidf = tfidf_vectorizer.transform([new_text_document])

# Transform the TF-IDF vector into LSA space (if you used LSA)
new_data_point_lsa = lsa.transform(new_data_point_tfidf)

# Predict the cluster of the new data point
predicted_cluster = kmeans_model.predict(new_data_point_lsa)

# Assuming kmeans_result contains the cluster assignments
data['cluster_label'] = kmeans_result

# Visualize the Predicted Cluster 
print(f"Predicted Cluster for New Data Point: {predicted_cluster[0]}")

# Visualize the clustering results within the clustering frame
plt.figure(figsize=(8, 6))
for cluster_index in range(optimal_k):
    index = np.where(kmeans_result == cluster_index)[0]
    x_values = lsa_matrix[index, 0]
    y_values = lsa_matrix[index, 1]
    plt.scatter(x_values, y_values, label=f'Cluster {cluster_index + 1}')

# Show the K-Means plot
plt.title('K-Means Clustering Results')
plt.xlabel('LSA Dimension 1')
plt.ylabel('LSA Dimension 2')
plt.show()
plt.legend()

# Silhouette analysis for clustering
silhouette_avg = silhouette_score(lsa_matrix, kmeans_result)
print(f"Average silhouette score: {silhouette_avg}")

# Create a new Tkinter window
window = tk.Tk()
window.title("Project Clustering and Search")

# Create a frame for search
search_frame = tk.Frame(window)
search_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create a frame for clustering plot
clustering_frame = tk.Frame(window)
clustering_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label for the Elbow Method plot
elbow_label = tk.Label(clustering_frame, text="Elbow Method Plot:")
elbow_label.pack()

# Create and place a search bar within the search frame
search_label = tk.Label(search_frame, text="Search:")
search_label.pack()
search_entry = tk.Entry(search_frame)
search_entry.pack()

# Create a function to update the search results
def update_search_results():
    user_query = search_entry.get().strip()
    if user_query:
        search_results = search_documents(user_query, lsa_matrix, data, kmeans_result)
        if search_results:
            search_output.delete(1.0, tk.END)  # Clear previous results
            search_output.insert(tk.END, f"Search results for '{user_query}':\n")
            for i, text in enumerate(search_results, start=1):
                search_output.insert(tk.END, f"{i}. {text}\n")
        else:
            search_output.delete(1.0, tk.END)
            search_output.insert(tk.END, "No matching documents found.")
    else:
        search_output.delete(1.0, tk.END)
        search_output.insert(tk.END, "Please enter a search query.")

# Create a search button within the search frame
search_button = tk.Button(search_frame, text="Search", command=update_search_results)
search_button.pack()

# Create a scrolled text widget for displaying search results within the search frame
search_output = scrolledtext.ScrolledText(search_frame, wrap=tk.WORD, width=40, height=10)
search_output.pack()

# Function to search for documents
def search_documents(query, lsa_matrix, data, kmeans_result):
    search_results = []
    for i in range(len(data)):
        if query.lower() in data['text_data'][i].lower():
            cluster_label = kmeans_result[i]
            text = data['text_data'][i]
            search_results.append(f"Cluster {cluster_label + 1}: {text}")
    return search_results

# Create a label for the Silhouette Score
silhouette_avg = silhouette_score(lsa_matrix, kmeans_result)
silhouette_label = tk.Label(clustering_frame, text=f"Average Silhouette Score: {silhouette_avg:.2f}")
silhouette_label.pack()

# Create a label for the Calinski-Harabasz Index
ch_score = calinski_harabasz_score(lsa_matrix, kmeans_result)
ch_label = tk.Label(clustering_frame, text=f"Calinski-Harabasz Index: {ch_score:.2f}")
ch_label.pack()

# Create a label for the Davies-Bouldin Index
db_score = davies_bouldin_score(lsa_matrix, kmeans_result)
db_label = tk.Label(clustering_frame, text=f"Davies-Bouldin Index: {db_score:.2f}")
db_label.pack()

window.mainloop()
