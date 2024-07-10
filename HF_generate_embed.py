import fitz  # PyMuPDF
import pdfplumber
import re
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModel

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

# Load Hugging Face model and tokenizer
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to extract and split paragraphs based on section headers
def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    section_pattern = re.compile(r'(\d+\.\d+)')
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        # Split text based on section headers
        raw_paragraphs = section_pattern.split(text)
        # Combine the header and the corresponding paragraph
        combined_paragraphs = [raw_paragraphs[i] + raw_paragraphs[i + 1] for i in range(1, len(raw_paragraphs) - 1, 2)]
        paragraphs.extend([p.strip() for p in combined_paragraphs if p.strip()])
    return paragraphs

# Function to extract tables from a PDF using pdfplumber
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                table_str = "\n".join(["\t".join([cell if cell is not None else "" for cell in row]) for row in table])
                tables.append(table_str)
    return tables

# Function to save paragraphs and tables in a single text file
def save_sections(paragraphs, tables, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, paragraph in enumerate(paragraphs):
            file.write(f"Paragraph {i + 1}:\n{paragraph}\n\n")
        for i, table in enumerate(tables):
            file.write(f"Table {i + 1}:\n{table}\n\n")

# Function to save embeddings in a separate text file
def save_embeddings(paragraph_embeddings, table_embeddings, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, embedding in enumerate(paragraph_embeddings):
            file.write(f"Paragraph {i + 1} Embedding:\n{embedding.tolist()}\n\n")
        for i, embedding in enumerate(table_embeddings):
            file.write(f"Table {i + 1} Embedding:\n{embedding.tolist()}\n\n")

# Function to generate embeddings using Hugging Face model
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.array(embeddings)

def main(input_pdf, sections_output_file, embeddings_output_file):
    paragraphs = extract_paragraphs_from_pdf(input_pdf)
    tables = extract_tables_from_pdf(input_pdf)
    
    # Generate embeddings using Hugging Face model
    paragraph_embeddings = generate_embeddings(paragraphs)
    table_embeddings = generate_embeddings(tables)

    # Save sections and embeddings
    save_sections(paragraphs, tables, sections_output_file)
    save_embeddings(paragraph_embeddings, table_embeddings, embeddings_output_file)

# Usage
if __name__ == "__main__":
    input_pdf = 'Academic-Regulations.pdf'  # Path to your input PDF
    sections_output_file = 'sectionshug1.txt'  # Path to save the paragraphs and tables
    embeddings_output_file = 'embeddingshug1.txt'  # Path to save the embeddings
    main(input_pdf, sections_output_file, embeddings_output_file)
