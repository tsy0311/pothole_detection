"""
Direct PDF Reading API - Read PDF content without extracting files.

This module provides functions to read PDF content directly in memory.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class PDFReader:
    """Direct PDF reader that doesn't extract files to disk."""
    
    def __init__(self, pdf_path: str):
        """Initialize PDF reader with path to PDF file."""
        self.pdf_path = pdf_path
        self.doc = None
    
    def __enter__(self):
        """Context manager entry."""
        self.doc = fitz.open(self.pdf_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.doc:
            self.doc.close()
    
    def get_page_count(self) -> int:
        """Get total number of pages in PDF."""
        return len(self.doc) if self.doc else 0
    
    def read_page(self, page_num: int) -> str:
        """
        Read text content from a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            Text content of the page
        """
        if not self.doc:
            raise ValueError("PDF not opened. Use context manager or open()")
        
        if page_num < 0 or page_num >= len(self.doc):
            raise IndexError(f"Page {page_num} out of range (0-{len(self.doc)-1})")
        
        page = self.doc[page_num]
        return page.get_text()
    
    def read_all_pages(self) -> List[str]:
        """Read text content from all pages."""
        if not self.doc:
            raise ValueError("PDF not opened. Use context manager or open()")
        
        return [self.doc[i].get_text() for i in range(len(self.doc))]
    
    def search(self, search_term: str) -> List[Dict]:
        """
        Search for a term across all pages.
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of dictionaries with page number and context
        """
        if not self.doc:
            raise ValueError("PDF not opened. Use context manager or open()")
        
        results = []
        search_lower = search_term.lower()
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            if search_lower in text.lower():
                # Get lines containing the search term
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if search_lower in line.lower():
                        # Get context (2 lines before and after)
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        context = '\n'.join(lines[context_start:context_end])
                        
                        results.append({
                            'page': page_num + 1,
                            'line_number': i + 1,
                            'line': line.strip(),
                            'context': context
                        })
        
        return results
    
    def get_images_info(self, page_num: Optional[int] = None) -> List[Dict]:
        """
        Get information about images in PDF.
        
        Args:
            page_num: Specific page number (None for all pages)
            
        Returns:
            List of image information dictionaries
        """
        if not self.doc:
            raise ValueError("PDF not opened. Use context manager or open()")
        
        pages_to_check = [page_num] if page_num is not None else range(len(self.doc))
        images_info = []
        
        for pg_num in pages_to_check:
            if pg_num < 0 or pg_num >= len(self.doc):
                continue
                
            page = self.doc[pg_num]
            image_list = page.get_images(full=True)
            
            if image_list:
                images_info.append({
                    'page': pg_num + 1,
                    'count': len(image_list),
                    'xrefs': [img[0] for img in image_list]
                })
        
        return images_info
    
    def extract_image_bytes(self, page_num: int, image_index: int = 0) -> Optional[bytes]:
        """
        Extract image bytes directly without saving to disk.
        
        Args:
            page_num: Page number (0-indexed)
            image_index: Index of image on the page
            
        Returns:
            Image bytes or None if not found
        """
        if not self.doc:
            raise ValueError("PDF not opened. Use context manager or open()")
        
        if page_num < 0 or page_num >= len(self.doc):
            return None
        
        page = self.doc[page_num]
        image_list = page.get_images(full=True)
        
        if image_index >= len(image_list):
            return None
        
        try:
            xref = image_list[image_index][0]
            base_image = self.doc.extract_image(xref)
            return base_image["image"]
        except Exception:
            return None


# Convenience functions for direct usage
def read_pdf_page(pdf_path: str, page_num: int) -> str:
    """Read a specific page from PDF."""
    with PDFReader(pdf_path) as reader:
        return reader.read_page(page_num)


def search_pdf(pdf_path: str, search_term: str) -> List[Dict]:
    """Search for term in PDF."""
    with PDFReader(pdf_path) as reader:
        return reader.search(search_term)


def read_pdf_all(pdf_path: str) -> List[str]:
    """Read all pages from PDF."""
    with PDFReader(pdf_path) as reader:
        return reader.read_all_pages()


# Example usage
if __name__ == "__main__":
    pdf_path = "H00390718_FinalDissertationReport.pdf"
    
    # Example 1: Read a specific page
    print("=" * 60)
    print("EXAMPLE 1: Reading Page 4 (Abstract)")
    print("=" * 60)
    abstract = read_pdf_page(pdf_path, 3)  # Page 4 (0-indexed)
    print(abstract[:500])  # First 500 characters
    print("...")
    
    # Example 2: Search for content
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Searching for 'confidence'")
    print("=" * 60)
    results = search_pdf(pdf_path, "confidence")
    for result in results[:3]:  # First 3 results
        print(f"\nPage {result['page']}:")
        print(result['context'])
    
    # Example 3: Using context manager for multiple operations
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multiple operations with context manager")
    print("=" * 60)
    with PDFReader(pdf_path) as reader:
        print(f"Total pages: {reader.get_page_count()}")
        print(f"Images found: {len(reader.get_images_info())} pages with images")
        
        # Read page 10
        page10 = reader.read_page(9)
        print(f"\nPage 10 preview (first 200 chars):")
        print(page10[:200])

