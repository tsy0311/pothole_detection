import PyPDF2
import sys
from pathlib import Path

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"Total pages: {num_pages}")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

if __name__ == "__main__":
    pdf_file = "H00390718_FinalDissertationReport.pdf"
    
    if not Path(pdf_file).exists():
        print(f"Error: PDF file '{pdf_file}' not found!")
        sys.exit(1)
    
    print(f"Extracting text from {pdf_file}...")
    extracted_text = extract_pdf_text(pdf_file)
    
    if extracted_text:
        # Save to text file
        output_file = "extracted_pdf_text.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"\nText extracted and saved to '{output_file}'")
        print(f"Total characters extracted: {len(extracted_text)}")
        
        # Display first 1000 characters as preview
        print("\n--- Preview (first 1000 characters) ---")
        print(extracted_text[:1000])
    else:
        print("Failed to extract text from PDF")

