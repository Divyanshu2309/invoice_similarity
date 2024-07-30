import ssl
import nltk
import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def extract_features(text):
    """
    Extract features like invoice number, date, and amount from the text.
    
    :param text: Preprocessed text.
    :return: Extracted features as a dictionary.
    """
    features = {
        'invoice_number': re.findall(r'invoice\s*#?\s*(\d+)', text),
        'date': re.findall(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', text),
        'amount': re.findall(r'\b\d+\.\d{2}\b', text)
    }
    return features

def calculate_structural_similarity(features1, features2):
    """
    Calculate structural similarity between two sets of features.
    
    :param features1: First set of features.
    :param features2: Second set of features.
    :return: Structural similarity score.
    """
    common_keys = set(features1.keys()).intersection(set(features2.keys()))
    total_keys = set(features1.keys()).union(set(features2.keys()))
    
    score = 0
    for key in common_keys:
        if features1[key] == features2[key]:
            score += 1
            
    return score / len(total_keys)

def main(input_pdf_path, database_pdfs):
    """
    Main function to process the input invoice and compare it with a database of invoices.
    
    :param input_pdf_path: Path to the input invoice PDF.
    :param database_pdfs: List of paths to database invoice PDFs.
    :return: Most similar invoice and similarity score.
    """
    input_text = preprocess_text(extract_text_from_pdf(input_pdf_path))
    input_features = extract_features(input_text)
    
    similarity_scores = []
    structural_scores = []
    
    for pdf in database_pdfs:
        db_text = preprocess_text(extract_text_from_pdf(pdf))
        db_features = extract_features(db_text)
        
        # Calculate text similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([input_text, db_text])
        text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Calculate structural similarity
        structural_similarity = calculate_structural_similarity(input_features, db_features)
        
        combined_score = (text_similarity + structural_similarity) / 2
        
        similarity_scores.append(combined_score)
        structural_scores.append(structural_similarity)
    
    most_similar_index = similarity_scores.index(max(similarity_scores))
    most_similar_invoice = database_pdfs[most_similar_index]
    similarity_score = similarity_scores[most_similar_index]
    structural_score = structural_scores[most_similar_index]
    
    return most_similar_invoice, similarity_score, structural_score

# Usage example
if __name__ == "__main__":
    input_pdf = '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/test/invoice_77098.pdf'
    database_pdfs = [
        '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/2024.03.15_1145.pdf',
        '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/Faller_8.PDF',
        '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/invoice_77073.pdf',
        '/Users/divyanshuagarwal/Desktop/invoice similarity/invoice_similarity/invoice_102856.pdf'
    ]
    most_similar_invoice, similarity_score, structural_score = main(input_pdf, database_pdfs)
    print(f'The most similar invoice is: {most_similar_invoice} with a similarity score of {similarity_score} and structural score of {structural_score}')
