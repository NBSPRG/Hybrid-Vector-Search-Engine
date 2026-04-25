from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample inputs
sample_inputs = [
    "How do I create a landing page?",
    "Best email marketing templates",
    "How to build an online store",
    "SEO optimization tips for blogs"
]

query = ["How do I make an e-commerce website?"]

# Vectorize and calculate similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sample_inputs + query)

similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]
best_match_idx = similarities.argmax()

print(f"Query: '{query[0]}'")
print(f"Top Match: '{sample_inputs[best_match_idx]}' (Score: {similarities[best_match_idx]:.2f})")
