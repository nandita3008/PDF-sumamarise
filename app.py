import os
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

print("Loading Hugging Face model and tokenizer...")
# Load Hugging Face model and tokenizer
hf_model_name = "BAAI/bge-m3"
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModel.from_pretrained(hf_model_name)
print("Hugging Face model and tokenizer loaded successfully.")

print("Loading summarization pipeline...")
# Load summarization pipeline
hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
print("Hugging Face summarization pipeline loaded successfully.")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to load sections and embeddings from files
def load_data(sections_file, embeddings_file):
    print(f"Loading sections from file {sections_file}...")
    with open(sections_file, 'r', encoding='utf-8') as file:
        sections = file.read().split('\n\n')
    print(f"Loaded {len(sections)} sections.")

    print(f"Loading embeddings from file {embeddings_file}...")
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
    return sections, embeddings

# Function to generate embeddings using Hugging Face model
def generate_hf_embeddings(texts):
    print(f"Generating embeddings for {len(texts)} texts using Hugging Face model...")
    embeddings = []
    for text in texts:
        inputs = hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = hf_model(**inputs)
        # Flatten the embedding to 1D by averaging over the sequence dimension
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embedding = embedding.flatten()  # Ensure each embedding is 1D
        embeddings.append(embedding)
    print("Embeddings generated successfully using Hugging Face model.")
    return np.array(embeddings)

# Function to generate embeddings using OpenAI model
def generate_openai_embeddings(texts):
    print(f"Generating embeddings for {len(texts)} texts using OpenAI model...")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embedding = np.array(response['data'][0]['embedding'])
        embeddings.append(embedding)
    print("Embeddings generated successfully using OpenAI model.")
    return np.array(embeddings)

# Function to find the top N closest sections to the question embedding
def find_top_sections(question_embedding, embeddings, sections, top_n=5):
    print("Finding top sections...")
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_sections = [(sections[i], similarities[i]) for i in top_indices]
    print(f"Top {top_n} sections found.")
    return top_sections

# Function to summarize an article using Hugging Face model
def summarize_hf_article(article, chunk_size=300, max_summary_length=300, min_summary_length=200):
    print("Summarizing article using Hugging Face model...")
    chunk_size = len(article) // 3
    chunks = [article[i:i + chunk_size] for i in range(0, len(article) - chunk_size, chunk_size)]
    chunks.append(article[4 * chunk_size:])  # Add the last chunk which includes any overflow

    summaries = []
    for chunk in chunks:
        if len(chunk) > 300:
            summary = hf_summarizer(chunk, max_length=max_summary_length, min_length=min_summary_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
    combined_summary = ' '.join(summaries)
    final_summary = hf_summarizer(combined_summary, max_length=500, min_length=400, do_sample=False)[0]['summary_text']
    print("Article summarized successfully using Hugging Face model.")
    return final_summary

# Function to provide an answer using OpenAI model
def answer_openai_question(question, article):
    print("Answering question using OpenAI model...")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Here is some context:\n\n{article}\n\nNow, answer the following question:\n\n{question}"}
        ]
    )
    answer = response['choices'][0]['message']['content'].strip()
    print("Question answered successfully using OpenAI model.")
    return answer

def main(question, model_type="hf", top_n=5):
    print(f"Received question: {question} with model type: {model_type}")
    
    # Load the appropriate files based on model type
    if model_type == "hf":
        sections, embeddings = load_data('sectionshug.txt', 'embeddingshug.txt')
        question_embedding = generate_hf_embeddings([question])[0]
    elif model_type == "openai":
        sections, embeddings = load_data('sectionsopen.txt', 'embeddingsopen.txt')
        question_embedding = generate_openai_embeddings([question])[0]
    else:
        raise ValueError("Invalid model type. Please choose 'hf' for Hugging Face or 'openai' for OpenAI.")

    # Find top N closest sections
    top_sections = find_top_sections(question_embedding, embeddings, sections, top_n)
    
    # Combine the top N sections into a single article
    article = ' '.join([section for section, similarity in top_sections])
    
    # Provide an answer based on the model type
    if model_type == "hf":
        summary = summarize_hf_article(article)
        answer = summary  # For HF, we continue to summarize
    elif model_type == "openai":
        answer = answer_openai_question(question, article)
    
    return answer

@app.route('/')
def index():
    print("Serving index page")
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    question = data.get('question', '')
    model_type = data.get('model_type', 'hf')  # Default to Hugging Face if not specified
    top_n = data.get('top_n', 5)
    
    print(f"Summarizing for question: {question} with model type: {model_type} and top_n: {top_n}")
    answer = main(question, model_type, top_n)
    print(f"Answer generated: {answer}")
    return jsonify({'summary': answer})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=False, port=5000)  # Make sure to run the server on port 5000
