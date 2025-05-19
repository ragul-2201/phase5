import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet
import time

#cutsomer segementation
def segment_customers(data):
    kmeans = KMeans(n_clusters=3)
    data['Segment'] = kmeans.fit_predict(data[['purchase_count', 'avg_spend']])
    return data

# Sample customer data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'purchase_count': [5, 20, 15, 10],
    'avg_spend': [200, 1500, 1200, 800]
})
segmented = segment_customers(customers)
print("Segmented Customers:\n", segmented)

# Recommendation Engine (Collaborative Filter)
def recommend(user_id, ratings_matrix):
    user_sim = cosine_similarity(ratings_matrix)
    sim_scores = user_sim[user_id]
    rec_scores = ratings_matrix.T.dot(sim_scores) / sim_scores.sum()
    return rec_scores

# Sample ratings matrix
ratings = pd.DataFrame({
    0: [5, 0, 3, 0],
    1: [4, 0, 0, 2],
    2: [0, 5, 4, 0],
}).T

recommendations = recommend(0, ratings.fillna(0).values)
print("Recommendations for User 0:\n", recommendations)

# Real-Time Offer Notification Example
def send_offer(customer_id, message):
    print(f"[Notification] Sent to Customer {customer_id}: {message}")

# Simulate a trigger
customer_action = {'customer_id': 1, 'action': 'visited_product_page'}
if customer_action['action'] == 'visited_product_page':
    send_offer(customer_action['customer_id'], "Special offer just for you!")

# Data Privacy with Encryption
key = Fernet.generate_key()
cipher = Fernet(key)

sensitive_data = "user_email@example.com"
encrypted = cipher.encrypt(sensitive_data.encode())
decrypted = cipher.decrypt(encrypted).decode()

print("Encrypted Data:", encrypted)
print("Decrypted Data:", decrypted)


# Performance Metrics Example
start_time = time.time()
time.sleep(0.5)  # Simulate processing time
end_time = time.time()
latency = end_time - start_time
print("Chatbot Latency: {:.3f} seconds".format(latency))



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# 1. Sample Customer Data (interactions + feedback)
data = {
    'User': ['Alice', 'Alice', 'Bob', 'Charlie', 'Charlie', 'Dave'],
    'Product': ['Laptop', 'Phone', 'Laptop', 'Phone', 'Tablet', 'Tablet'],
    'Rating': [5, 3, 4, 4, 5, 2],
    'Feedback': [
        "Love the laptop quality!",
        "Phone is okay.",
        "Good performance.",
        "Battery life is great!",
        "Very responsive and sleek.",
        "Too slow for my work."
    ]
}

df = pd.DataFrame(data)

# 2. Create User-Product Matrix
matrix = df.pivot_table(index='User', columns='Product', values='Rating').fillna(0)

# 3. Similarity Matrix
similarity = pd.DataFrame(
    cosine_similarity(matrix),
    index=matrix.index,
    columns=matrix.index
)

# 4. Recommend Products
def recommend(user, n=2):
    similar_users = similarity[user].sort_values(ascending=False).index[1:]
    owned = df[df['User'] == user]['Product'].tolist()
    scores = {}

    for sim_user in similar_users:
        sim_data = df[df['User'] == sim_user]
        for _, row in sim_data.iterrows():
            if row['Product'] not in owned:
                scores[row['Product']] = scores.get(row['Product'], 0) + row['Rating']
        if len(scores) >= n:
            break

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

# 5. Personalized Email Generator
def generate_email(user):
    recs = recommend(user)
    email = f"Hello {user},\n\nBased on your interest, we think you'll love:\n"
    for product, _ in recs:
        email += f"- {product}\n"
    email += "\nVisit our site to explore more personalized deals!\n"
    return email

# 6. Sentiment Analysis
def analyze_sentiment():
    df['Sentiment'] = df['Feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df[['User', 'Product', 'Feedback', 'Sentiment']]

# 7. Customer Segmentation
def segment_customers():
    sentiment_scores = analyze_sentiment().groupby('User')['Sentiment'].mean()
    segments = sentiment_scores.apply(lambda x: 'Promoter' if x > 0.5 else 'Passive' if x > 0 else 'Detractor')
    return segments

# === DEMO ===
print("=== Recommendations for Alice ===")
print(recommend('Alice'))

print("\n=== Personalized Email ===")
print(generate_email('Alice'))

print("\n=== Sentiment Analysis ===")
print(analyze_sentiment())

print("\n=== Customer Segmentation ===")
print(segment_customers())
