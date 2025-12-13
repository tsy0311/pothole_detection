import fitz  # PyMuPDF
import sys

def read_pdf_direct(pdf_path: str, show_text: bool = True, show_images: bool = False):
    """
    Read PDF directly without extracting to disk.
    
    Args:
        pdf_path: Path to PDF file
        show_text: Whether to display text content
        show_images: Whether to display image information
    """
    print(f"Reading PDF: {pdf_path}\n")
    print("=" * 80)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    print(f"Total pages: {len(doc)}")
    print("=" * 80)
    
    # Read content directly
    all_text = []
    image_info = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        if show_text:
            text = page.get_text()
            all_text.append(f"\n{'='*80}\nPAGE {page_num + 1}\n{'='*80}\n{text}\n")
        
        if show_images:
            image_list = page.get_images(full=True)
            if image_list:
                image_info.append({
                    'page': page_num + 1,
                    'count': len(image_list),
                    'images': image_list
                })
    
    doc.close()
    
    # Display text if requested
    if show_text:
        print("\n".join(all_text[:3]))  # Show first 3 pages
        if len(all_text) > 3:
            print(f"\n... ({len(all_text) - 3} more pages) ...")
    
    # Display image info if requested
    if show_images:
        print("\n" + "=" * 80)
        print("IMAGE INFORMATION:")
        print("=" * 80)
        for info in image_info:
            print(f"Page {info['page']}: {info['count']} image(s)")
    
    return all_text, image_info


def search_pdf_content(pdf_path: str, search_term: str):
    """Search for specific content in PDF."""
    doc = fitz.open(pdf_path)
    results = []
    
    print(f"\nSearching for: '{search_term}'")
    print("=" * 80)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if search_term.lower() in text.lower():
            # Find context around the search term
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if search_term.lower() in line.lower():
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    context = '\n'.join(lines[context_start:context_end])
                    results.append({
                        'page': page_num + 1,
                        'context': context,
                        'line': line.strip()
                    })
                    break
    
    doc.close()
    
    for result in results:
        print(f"\nPage {result['page']}:")
        print("-" * 80)
        print(result['context'])
        print()
    
    return results


if __name__ == "__main__":
    pdf_path = "H00390718_FinalDissertationReport.pdf"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--search" and len(sys.argv) > 2:
            search_pdf_content(pdf_path, sys.argv[2])
        elif sys.argv[1] == "--images":
            read_pdf_direct(pdf_path, show_text=False, show_images=True)
        elif sys.argv[1] == "--full":
            read_pdf_direct(pdf_path, show_text=True, show_images=True)
        else:
            print("Usage:")
            print("  python read_pdf_direct.py              # Read first few pages")
            print("  python read_pdf_direct.py --full       # Show text and image info")
            print("  python read_pdf_direct.py --images     # Show only image info")
            print("  python read_pdf_direct.py --search TERM  # Search for term")
    else:
        # Default: read first few pages
        read_pdf_direct(pdf_path, show_text=True, show_images=False)

