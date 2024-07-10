import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
from flask import Flask, request, jsonify, render_template

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

print("Loading Hugging Face model and tokenizer...")
# Load Hugging Face model and tokenizer
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")

print("Loading summarization pipeline...")
# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
print("Summarization pipeline loaded successfully.")

app = Flask(__name__)

# Load sections and embeddings during server startup
sections_file = 'sectionshug.txt'
embeddings_file = 'embeddingshug.txt'

print("Loading sections from file...")
with open(sections_file, 'r', encoding='utf-8') as file:
    sections = file.read().split('\n\n')
print(f"Loaded {len(sections)} sections.")

print("Loading embeddings from file...")
embeddings = []
with open(embeddings_file, 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith("Paragraph") or line.startswith("Table"):
            continue
        if line.strip():
            embedding = np.array(eval(line.strip()))
            embeddings.append(embedding)
embeddings = np.array(embeddings)
print(f"Loaded {len(embeddings)} embeddings.")

# Reshape embeddings to remove extra dimension if present
if embeddings.ndim == 3:
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[2]))
print("Embeddings reshaped successfully.")

# Function to generate embeddings using Hugging Face model
def generate_embeddings(texts):
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        # Flatten the embedding to 1D by averaging over the sequence dimension
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embedding = embedding.flatten()  # Ensure each embedding is 1D
        embeddings.append(embedding)
    print("Embeddings generated successfully.")
    return np.array(embeddings)

# Function to find the top N closest sections to the question embedding
def find_top_sections(question_embedding, embeddings, sections, top_n=5):
    print("Finding top sections...")
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_sections = [(sections[i], similarities[i]) for i in top_indices]
    print(f"Top {top_n} sections found.")
    return top_sections

def summarize_article(article, chunk_size=300, max_summary_length=300, min_summary_length=200):
    print("Summarizing article...")
    chunk_size = len(article) // 3
    chunks = [article[i:i + chunk_size] for i in range(0, len(article) - chunk_size, chunk_size)]
    chunks.append(article[4 * chunk_size:])  # Add the last chunk which includes any overflow

    summaries = []
    for chunk in chunks:
        if len(chunk) > 300:
            summary = summarizer(chunk, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
    combined_summary = ' '.join(summaries)
    final_summary = summarizer(combined_summary, max_length=500, min_length=400, do_sample=False)[0]['summary_text']
    print("Article summarized successfully.")
    return final_summary

def main(question, top_n=5):
    print(f"Received question: {question}")
    # Create embedding for the question using Hugging Face model
    question_embedding = generate_embeddings([question])[0]
    
    # Find top N closest sections
    top_sections = find_top_sections(question_embedding, embeddings, sections, top_n)
    
    # Combine the top N sections into a single article
    article = ' '.join([section for section, similarity in top_sections])
    
    # Summarize the combined article
    summary = summarize_article(article)
    
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    question = data.get('question', '')
    top_n = data.get('top_n', 5)
    
    print(f"Summarizing for question: {question} with top_n: {top_n}")
    summary = main(question, top_n)
    print(f"Summary generated: {summary}")
    return jsonify({'summary': summary})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=False)
