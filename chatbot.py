import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

# Load dataset
df = pd.read_csv("data/crime_dataset.csv")

# Load a more robust Sentence Transformer model
print("Loading model...")
model = SentenceTransformer("all-mpnet-base-v2")

# Encode complaints once
complaints = df["Example Complaint (Query)"].tolist()
complaint_embeddings = model.encode(complaints, convert_to_tensor=True)

def extract_keywords(text):
    """
    Simple keyword extraction by splitting sentences and removing trivial words.
    """
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    words = text.split()
    # Remove common stopwords (optional: you can add more)
    stopwords = ["the", "a", "an", "in", "on", "at", "of", "and", "is", "was", "after", "with"]
    keywords = [w for w in words if w not in stopwords]
    return [" ".join(keywords)]  # Return as a single string for now

def chatbot(query, top_k=3, threshold=0.4):
    # Extract keywords or key phrases from the query
    phrases = extract_keywords(query)

    all_results = []
    for phrase in phrases:
        # Encode the phrase
        query_embedding = model.encode(phrase, convert_to_tensor=True)
        # Compute similarity
        scores = util.cos_sim(query_embedding, complaint_embeddings)[0]
        top_results = scores.topk(k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            if score.item() >= threshold:
                row = df.iloc[idx.item()]
                all_results.append({
                    "similarity": round(score.item(), 2),
                    "ipc": row["IPC Section(s)"],
                    "bns": row["BNS Section(s)"],
                    "explanation": row["Explanation (Plain)"],
                    "punishment": row["Punishment"]
                })

    # Remove duplicates by IPC + BNS combination
    seen = set()
    unique_results = []
    for res in all_results:
        key = (res["ipc"], res["bns"])
        if key not in seen:
            unique_results.append(res)
            seen.add(key)

    return unique_results

# Console chatbot loop
if __name__ == "__main__":
    print("⚖️ Crime Chatbot (IPC ↔ BNS Mapping)\nType 'exit' to quit.\n")
    while True:
        user_input = input("Enter your complaint: ")
        if user_input.lower() == "exit":
            print("Chatbot closed.")
            break

        matches = chatbot(user_input)

        if not matches:
            print("\n❌ Sorry, No relevant section found.\n")
        else:
            print("\n📌 Possible Matched Sections:")
            for i, res in enumerate(matches, 1):
                print(f"{i}. (Similarity: {res['similarity']})")
                print(f"   IPC: {res['ipc']}")
                print(f"   BNS: {res['bns']}")
                print(f"   Explanation: {res['explanation']}")
                print(f"   Punishment: {res['punishment']}\n")
