# Invoice_similarity
It is a program that takes an input invoice (PDF format) and compares it to a database of existing invoices.It calculates similarity scores based on text content and structural features.

# Approach
## Document Representation Method
1. The text content of each PDF invoice is extracted using PyMuPDF.
2. Then the extracted text is preprocessed by connverting to lowercase, removing special characters, and removing stopwords.
3. The preprocessed text is then used to extract features like invoice number, date, and amount are extracted using regular expressions.
4. This text is also then used to create some feature vectors of TF-IDF (Term Frequency-Inverse Document Frequency).

## Similarity Metric:
Now we have to go with the similarity calculation for this I have used the cosine similarity for Text Similarity and a score is calculated in Structural Similarity
### Text Similarity:
TF-IDF vectors are created for the preprocessed text, and cosine similarity is calculated to measure the textual similarity between documents.
### Structural Similarity:
Structural features are compared, and a score is calculated based on the presence of common features. This score is combined with the text similarity score to provide an overall similarity score.

# Instructions on How to Run the Code and Use the Provided Data
## Installation:

1. Ensure you have PythoN is installed on your system.
2. Install the required Python libraries:
    ```
   pip install pymupdf nltk scikit-learn
    ```
3. Download necessary NLTK data:
   ```
   import ssl
   import nltk
   try:
     _create_unverified_https_context = ssl._create_unverified_context
   except AttributeError:
      pass
   else:
       ssl._create_default_https_context = _create_unverified_https_context
   nltk.download('stopwords')
   nltk.download('punkt')
   ```
## Setup:
1. Place your input invoice PDF and database invoices in the appropriate directories.
2. Update the input_pdf and database_pdfs paths in the script to point to your invoice files.

```
input_pdf = '/path/to/your/input/invoice.pdf'
database_pdfs = [
    '/path/to/your/database/invoice1.pdf',
    '/path/to/your/database/invoice2.pdf',
    '/path/to/your/database/invoice3.pdf'
]
most_similar_invoice, similarity_score, structural_score = main(input_pdf, database_pdfs)
print(f'The most similar invoice is: {most_similar_invoice} with a similarity score of {similarity_score} and structural score of {structural_score}')
```

## Running the Script:
1. Run the script:
   ```
   python3 invoice_similarity.py
   ```
2. The script will output the most similar invoice along with the similarity score and structural score. 
