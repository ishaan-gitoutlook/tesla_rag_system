"""
data_loader.py
Modular functions for PDF ingestion, cleaning, and text chunking.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
import pypdf


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page by page from a PDF file.
    
    Args:
        pdf_path: Path to the target PDF file.
        
    Returns:
        List of dicts containing page_number (1-indexed) and raw text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    reader = pypdf.PdfReader(str(path))
    pages = []
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = page.extract_text() or ""
        pages.append({
            "page_number": page_num,
            "text": text
        })
        
    return pages


def clean_text(text: str) -> str:
    """
    Normalizes whitespace and standardizes table/text formatting.
    Merges broken table rows and isolated characters while preserving paragraph structure.
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned text string.
    """
    text = re.sub(r'[\t\r\f\v]+', ' ', text)
    text = text.replace('\xa0', ' ')
    
    # Process lines to merge table columns and broken lines cleanly
    raw_lines = [line.strip() for line in text.splitlines()]
    merged_lines = []
    buf = []
    
    for line in raw_lines:
        if not line:
            if buf:
                merged_lines.append(" ".join(buf))
                buf = []
            merged_lines.append("")
        else:
            # If line is short (e.g. $, numbers, percentages, symbols) or buf ends with short token, combine
            if len(line) <= 15 or (buf and len(buf[-1]) <= 15):
                buf.append(line)
            else:
                if buf:
                    merged_lines.append(" ".join(buf))
                    buf = []
                buf.append(line)
                
    if buf:
        merged_lines.append(" ".join(buf))
        
    text = "\n".join(merged_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(
    pages: List[Dict[str, Any]], 
    chunk_size: int = 750, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Chunks pages into overlapping segments with rich metadata.
    
    Args:
        pages: List of dicts with page_number and raw text.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        
    Returns:
        List of chunk dicts with chunk_id, page_number, text, and char_count.
    """
    chunks = []
    chunk_counter = 0

    for page_data in pages:
        page_num = page_data["page_number"]
        cleaned = clean_text(page_data["text"])
        
        if not cleaned:
            continue
            
        start = 0
        text_length = len(cleaned)
        
        while start < text_length:
            end = start + chunk_size
            
            # If we are not at the end of the text, try to break at a newline or space
            if end < text_length:
                boundary = cleaned.rfind('\n', start + chunk_size // 2, end)
                if boundary == -1:
                    boundary = cleaned.rfind('. ', start + chunk_size // 2, end)
                    if boundary != -1:
                        boundary += 1
                if boundary == -1:
                    boundary = cleaned.rfind(' ', start + chunk_size // 2, end)
                if boundary != -1 and boundary > start:
                    end = boundary
            
            chunk_content = cleaned[start:end].strip()
            
            if len(chunk_content) > 30:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter}",
                    "page_number": page_num,
                    "text": chunk_content,
                    "char_count": len(chunk_content)
                })
                chunk_counter += 1
                
            start = end - chunk_overlap if (end - chunk_overlap) > start else end

    return chunks


def load_and_chunk_pdf(
    pdf_path: str, 
    chunk_size: int = 750, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Convenience modular function to load PDF and return chunked documents.
    """
    pages = extract_text_from_pdf(pdf_path)
    return chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
