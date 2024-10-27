from typing import List
from llama_index.core.schema import (Document)

def add_metadata_to_documents(documents: List[Document]) -> List[Document]:
    for document in documents:
        original_metadata = document.metadata

        additional_metadata = {
            "total_pages_in_original_pdf": len(documents),
            "size_of_original_pdf(MB)": f"{original_metadata.get('file_size') / (1024*1024):.2f} MB"
        }

        document.metadata = original_metadata | additional_metadata
    
    return documents

