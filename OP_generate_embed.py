import fitz  # PyMuPDF
import pdfplumber
import re
import openai
import numpy as np
import warnings
import os

# Set your OpenAI API key here
openai.api_key = os.getenv('OPENAI_API_KEY')

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

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

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model).data[0].embedding

# Function to generate embeddings using OpenAI
def generate_embeddings(texts):
    embeddings = []
    text_list = []
    for text in texts:
        try:
            embedding = get_embedding(text, model='text-embedding-ada-002')
            text_list.append(text)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Failed to generate embedding for text: {text}\nError: {e}")
            embeddings.append(("Failed", []))
            pass

    return np.array(embeddings)

def main(input_pdf, sections_output_file, embeddings_output_file):
    paragraphs = extract_paragraphs_from_pdf(input_pdf)
    tables = extract_tables_from_pdf(input_pdf)
    
    # For this example, we are not using tables for embeddings
    tables = []

    # Generate embeddings using OpenAI
    paragraph_embeddings = generate_embeddings(paragraphs)
    table_embeddings = []

    # Combine paragraphs and tables
    combined_sections = paragraphs
    combined_embeddings = paragraph_embeddings

    # Save sections and embeddings
    save_sections(paragraphs, tables, sections_output_file)
    save_embeddings(paragraph_embeddings, table_embeddings, embeddings_output_file)

# Usage
if __name__ == "__main__":
    input_pdf = 'Academic-Regulations.pdf'  # Path to your input PDF
    sections_output_file = 'sectionsopen1.txt'  # Path to save the paragraphs and tables
    embeddings_output_file = 'embeddingsopen1.txt'  # Path to save the embeddings
    main(input_pdf, sections_output_file, embeddings_output_file)
