import ssl
import nltk

# Workaround to handle SSL certificate verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters,
    and removing stopwords.
    
    :param text: Raw text.
    :return: Processed text.
    """
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def calculate_similarity(texts):
    """
    Calculate similarity scores between documents using TF-IDF and Cosine Similarity.
    
    :param texts: List of texts to compare.
    :return: Similarity matrix.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

def main(input_pdf_path, database_pdfs):
    """
    Main function to process the input invoice and compare it with a database of invoices.
    
    :param input_pdf_path: Path to the input invoice PDF.
    :param database_pdfs: List of paths to database invoice PDFs.
    :return: Most similar invoice and similarity score.
    """
    input_text = preprocess_text(extract_text_from_pdf(input_pdf_path))
    database_texts = [preprocess_text(extract_text_from_pdf(pdf)) for pdf in database_pdfs]
    
    all_texts = [input_text] + database_texts
    similarity_matrix = calculate_similarity(all_texts)
    
    similarities = similarity_matrix[0, 1:]
    most_similar_index = np.argmax(similarities)
    most_similar_invoice = database_pdfs[most_similar_index]
    similarity_score = similarities[most_similar_index]
    
    return most_similar_invoice, similarity_score

# Usage example
input_pdf = '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/test/invoice_77098.pdf'
database_pdfs = [
    '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/2024.03.15_1145.pdf',
    '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/Faller_8.PDF',
    '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/invoice_77073.pdf'
]
most_similar_invoice, similarity_score = main(input_pdf, database_pdfs)
print(f'The most similar invoice is: {most_similar_invoice} with a similarity score of {similarity_score}')
