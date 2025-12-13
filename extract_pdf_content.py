import fitz  # PyMuPDF
import os

pdf_path = "H00390718_FinalDissertationReport.pdf"
output_dir = "extracted_pdf_content"
images_dir = os.path.join(output_dir, "images")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Open PDF
doc = fitz.open(pdf_path)

# Extract text and images
text_content = []
image_count = 0

print(f"Total pages: {len(doc)}")
print("Extracting text and images...\n")

for page_num in range(len(doc)):
    page = doc[page_num]
    
    # Extract text
    text = page.get_text()
    text_content.append(f"\n{'='*80}\nPAGE {page_num + 1}\n{'='*80}\n{text}\n")
    
    # Extract images
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = os.path.join(images_dir, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            image_count += 1
            print(f"Extracted image: {image_filename}")
        except Exception as e:
            print(f"Error extracting image on page {page_num + 1}, img {img_index + 1}: {e}")

# Save all text
text_path = os.path.join(output_dir, "extracted_text.txt")
with open(text_path, "w", encoding="utf-8") as f:
    f.write("".join(text_content))

print(f"\n{'='*80}")
print(f"Extraction complete!")
print(f"Text saved to: {text_path}")
print(f"Total images extracted: {image_count}")
print(f"Images saved to: {images_dir}")
print(f"{'='*80}")

doc.close()

