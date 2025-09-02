#!/usr/bin/env python3
"""
PDF Processing Module for Arvocap Chatbot Training
Handles extraction and processing of text from PDF documents
"""

import os
import re
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import json

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, falling back to other methods")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logging.warning("pypdf not available")

from text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Advanced PDF processing class that extracts and structures content from PDF files
    Now with OCR support for scanned documents
    """
    
    def __init__(self, use_ocr: bool = True):
        self.text_processor = TextProcessor()
        self.supported_extensions = ['.pdf']
        self.use_ocr = use_ocr
        
        # Check OCR availability
        self.ocr_available = self._check_ocr_availability()
        
        logger.info(f"PDF Processor initialized (OCR: {'Available' if self.ocr_available else 'Not Available'})")
    
    def _check_ocr_availability(self) -> bool:
        """Check if OCR libraries are available"""
        try:
            import pytesseract
            import pdf2image
            from PIL import Image
            return True
        except ImportError:
            if self.use_ocr:
                logger.warning("OCR libraries not available. Install with: pip install pytesseract pdf2image Pillow")
            return False
        
    def extract_text_pymupdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using PyMuPDF (most reliable method)"""
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'page_count': len(doc)
            }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only add pages with content
                    pages_content.append({
                        'page': page_num + 1,
                        'text': text.strip(),
                        'char_count': len(text.strip())
                    })
            
            doc.close()
            
            return {
                'method': 'PyMuPDF',
                'metadata': metadata,
                'pages': pages_content,
                'total_text': '\n\n'.join([p['text'] for p in pages_content])
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return None
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using pdfplumber (good for tables and complex layouts)"""
        try:
            pages_content = []
            metadata = {}
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata = {
                    'page_count': len(pdf.pages),
                    'metadata': pdf.metadata or {}
                }
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    
                    if text and text.strip():
                        page_content = {
                            'page': page_num + 1,
                            'text': text.strip(),
                            'char_count': len(text.strip()),
                            'tables_count': len(tables) if tables else 0
                        }
                        
                        # Add table content as text
                        if tables:
                            table_text = self.convert_tables_to_text(tables)
                            page_content['text'] += f"\n\nTables:\n{table_text}"
                        
                        pages_content.append(page_content)
            
            return {
                'method': 'pdfplumber',
                'metadata': metadata,
                'pages': pages_content,
                'total_text': '\n\n'.join([p['text'] for p in pages_content])
            }
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return None
    
    def extract_text_pypdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using pypdf (fallback method)"""
        try:
            pages_content = []
            
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                metadata = {
                    'page_count': len(reader.pages),
                    'metadata': reader.metadata or {}
                }
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    
                    if text and text.strip():
                        pages_content.append({
                            'page': page_num + 1,
                            'text': text.strip(),
                            'char_count': len(text.strip())
                        })
            
            return {
                'method': 'pypdf',
                'metadata': metadata,
                'pages': pages_content,
                'total_text': '\n\n'.join([p['text'] for p in pages_content])
            }
            
        except Exception as e:
            logger.error(f"pypdf extraction failed for {pdf_path}: {e}")
            return None
    
    def convert_tables_to_text(self, tables: List) -> str:
        """Convert extracted tables to readable text format"""
        table_texts = []
        
        for i, table in enumerate(tables):
            if not table:
                continue
                
            table_text = f"Table {i+1}:\n"
            for row in table:
                if row:
                    # Filter out None values and join with tabs
                    clean_row = [str(cell) if cell is not None else '' for cell in row]
                    table_text += '\t'.join(clean_row) + '\n'
            
            table_texts.append(table_text)
        
        return '\n'.join(table_texts)
    
    def process_pdf(self, pdf_path: str) -> Optional[Dict[str, any]]:
        """
        Process a single PDF file using the best available method
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Try methods in order of preference
        methods = []
        if PYMUPDF_AVAILABLE:
            methods.append(('PyMuPDF', self.extract_text_pymupdf))
        if PDFPLUMBER_AVAILABLE:
            methods.append(('pdfplumber', self.extract_text_pdfplumber))
        if PYPDF_AVAILABLE:
            methods.append(('pypdf', self.extract_text_pypdf))
        
        if not methods:
            logger.error("No PDF processing libraries available!")
            return None
        
        result = None
        for method_name, method_func in methods:
            try:
                result = method_func(pdf_path)
                if result and result.get('total_text', '').strip():
                    logger.info(f"Successfully extracted text using {method_name}")
                    break
            except Exception as e:
                logger.warning(f"{method_name} failed for {pdf_path}: {e}")
                continue
        
        if not result:
            logger.error(f"All PDF extraction methods failed for {pdf_path}")
            return None
        
        # Process and clean the extracted text
        processed_result = self.post_process_content(result, pdf_path)
        return processed_result
    
    def post_process_content(self, extraction_result: Dict, pdf_path: str) -> Dict[str, any]:
        """
        Clean and structure the extracted PDF content
        """
        total_text = extraction_result.get('total_text', '')
        
        # Clean the text
        cleaned_text = self.text_processor.clean_text(total_text)
        
        # Extract key sections
        sections = self.extract_sections(cleaned_text)
        
        # Generate chunks for training
        chunks = self.text_processor.chunk_text(cleaned_text, chunk_size=1000, overlap=200)
        
        # Extract metadata
        filename = os.path.basename(pdf_path)
        file_metadata = {
            'filename': filename,
            'filepath': pdf_path,
            'file_size': os.path.getsize(pdf_path),
            'extraction_method': extraction_result.get('method', 'unknown'),
            'processed_at': self.text_processor.get_timestamp()
        }
        
        return {
            'source': pdf_path,
            'type': 'pdf',
            'metadata': {
                **file_metadata,
                **extraction_result.get('metadata', {})
            },
            'content': {
                'raw_text': total_text,
                'cleaned_text': cleaned_text,
                'sections': sections,
                'chunks': chunks,
                'page_info': extraction_result.get('pages', [])
            },
            'stats': {
                'total_pages': len(extraction_result.get('pages', [])),
                'total_chars': len(cleaned_text),
                'total_words': len(cleaned_text.split()),
                'total_chunks': len(chunks)
            }
        }
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract sections from the PDF text based on common patterns
        """
        sections = []
        
        # Common section patterns for financial/investment documents
        section_patterns = [
            r'(?i)(executive\s+summary|summary)',
            r'(?i)(introduction|overview)',
            r'(?i)(investment\s+strategy|strategy)',
            r'(?i)(portfolio\s+performance|performance)',
            r'(?i)(risk\s+factors|risks?)',
            r'(?i)(fund\s+information|fund\s+details)',
            r'(?i)(fees?\s+and\s+expenses|costs?)',
            r'(?i)(conclusion|summary)',
        ]
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        current_section = None
        section_content = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this paragraph starts a new section
            is_section_header = False
            for pattern in section_patterns:
                if re.match(pattern, paragraph):
                    # Save previous section
                    if current_section and section_content:
                        sections.append({
                            'title': current_section,
                            'content': '\n\n'.join(section_content)
                        })
                    
                    # Start new section
                    current_section = paragraph
                    section_content = []
                    is_section_header = True
                    break
            
            if not is_section_header:
                section_content.append(paragraph)
        
        # Add the last section
        if current_section and section_content:
            sections.append({
                'title': current_section,
                'content': '\n\n'.join(section_content)
            })
        
        return sections
    
    def process_pdf_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Process all PDF files in a directory
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        pdf_files = list(directory.glob('**/*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        processed_pdfs = []
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(str(pdf_file))
                if result:
                    processed_pdfs.append(result)
                    logger.info(f"Successfully processed: {pdf_file.name}")
                else:
                    logger.warning(f"Failed to process: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        return processed_pdfs
    
    def save_processed_content(self, processed_pdfs: List[Dict], output_file: str):
        """
        Save processed PDF content to JSON file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_pdfs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved processed PDF content to {output_file}")
            
            # Print summary
            total_pdfs = len(processed_pdfs)
            total_pages = sum(pdf.get('stats', {}).get('total_pages', 0) for pdf in processed_pdfs)
            total_chunks = sum(pdf.get('stats', {}).get('total_chunks', 0) for pdf in processed_pdfs)
            
            logger.info(f"PDF Processing Summary:")
            logger.info(f"  - Total PDFs processed: {total_pdfs}")
            logger.info(f"  - Total pages: {total_pages}")
            logger.info(f"  - Total training chunks: {total_chunks}")
            
        except Exception as e:
            logger.error(f"Error saving processed content: {e}")
    
    def extract_text_with_ocr(self, pdf_path: str) -> Dict[str, any]:
        """Extract text from PDF using OCR for scanned documents"""
        if not self.ocr_available:
            logger.warning("OCR not available, skipping OCR extraction")
            return None
        
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
            
            logger.info(f"Starting OCR extraction for: {pdf_path}")
            
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path)
            
            pages_content = []
            total_text = ""
            
            for page_num, page_image in enumerate(pages):
                try:
                    # Extract text from image using OCR
                    ocr_text = pytesseract.image_to_string(page_image, lang='eng')
                    
                    if ocr_text.strip():
                        page_content = {
                            'page': page_num + 1,
                            'text': ocr_text.strip(),
                            'char_count': len(ocr_text.strip()),
                            'extraction_method': 'OCR'
                        }
                        pages_content.append(page_content)
                        total_text += ocr_text + "\n\n"
                        
                    logger.info(f"OCR completed for page {page_num + 1}")
                    
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num + 1}: {e}")
                    continue
            
            metadata = {
                'page_count': len(pages),
                'extraction_method': 'OCR',
                'ocr_used': True
            }
            
            return {
                'method': 'OCR',
                'metadata': metadata,
                'pages': pages_content,
                'total_text': total_text.strip()
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return None
    
    def process_pdf_with_ocr(self, pdf_path: str) -> Optional[Dict[str, any]]:
        """
        Process PDF with OCR fallback for scanned documents
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        logger.info(f"Processing PDF with OCR support: {pdf_path}")
        
        # First try standard text extraction
        result = self.process_pdf(pdf_path)
        
        # Check if we got meaningful text
        if result and result.get('content', {}).get('cleaned_text', '').strip():
            text_length = len(result['content']['cleaned_text'].strip())
            if text_length > 100:  # If we have substantial text
                logger.info(f"Standard extraction successful ({text_length} chars)")
                # Mark that OCR was not needed
                result['stats']['ocr_used'] = False
                result['stats']['extraction_method'] = result.get('metadata', {}).get('extraction_method', 'standard')
                return result
        
        # If standard extraction failed or produced minimal text, try OCR
        if self.use_ocr and self.ocr_available:
            logger.info("Standard extraction produced minimal text, trying OCR...")
            ocr_result = self.extract_text_with_ocr(pdf_path)
            
            if ocr_result and ocr_result.get('total_text', '').strip():
                logger.info(f"OCR extraction successful")
                # Process OCR result
                processed_result = self.post_process_content(ocr_result, pdf_path)
                # Mark that OCR was used
                processed_result['stats']['ocr_used'] = True
                processed_result['stats']['extraction_method'] = 'OCR'
                return processed_result
        
        # Return whatever we got, even if minimal
        if result:
            result['stats']['ocr_used'] = False
            result['stats']['extraction_method'] = result.get('metadata', {}).get('extraction_method', 'standard')
            return result
        
        logger.error(f"All extraction methods failed for {pdf_path}")
        return None

def main():
    """
    Command-line interface for PDF processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF files for chatbot training")
    parser.add_argument('input_path', help='Path to PDF file or directory containing PDFs')
    parser.add_argument('--output', '-o', default='data/processed_pdfs.json', 
                       help='Output file for processed content')
    
    args = parser.parse_args()
    
    processor = PDFProcessor()
    
    if os.path.isfile(args.input_path):
        # Process single PDF
        result = processor.process_pdf(args.input_path)
        if result:
            processor.save_processed_content([result], args.output)
        else:
            logger.error("Failed to process PDF file")
    elif os.path.isdir(args.input_path):
        # Process directory of PDFs
        results = processor.process_pdf_directory(args.input_path)
        if results:
            processor.save_processed_content(results, args.output)
        else:
            logger.error("No PDF files were successfully processed")
    else:
        logger.error(f"Invalid input path: {args.input_path}")

if __name__ == "__main__":
    main()
