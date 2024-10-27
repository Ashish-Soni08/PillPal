from typing import List

from dotenv import dotenv_values

from llama_index.core import (SimpleDirectoryReader, 
                              Settings,
                              VectorStoreIndex)

from llama_index.core.ingestion import IngestionPipeline

from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.schema import (BaseNode,
                                     Document,
                                     MetadataMode)

from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_parse import LlamaParse

from qdrant_client import QdrantClient

from nvidia_nims import (embedding_model,
                         llm,
                         rerank_model)

from utils import add_metadata_to_documents

config = dotenv_values(".env")

def extract(pdf_document: str = ["./sample_data/ozempic.pdf"], language: str = "en", target_pages: str = None):

    parsing_instructions = """
    The provided document is a thin piece of folded paper that is part of every drug prescription box. 
    Usually the text is in VERY small print and typically provides information about dosages, side effects, storage instructions and much more. 
    Try to extract the key information so that it is easy to understand.
    """

    pdf_parser = LlamaParse(
    api_key=config["LLAMACLOUD_API_KEY"],
    result_type="text",  # markdown doesn't work with fast_mode to True
    parsing_instruction=parsing_instructions,
    num_workers=7,
    check_interval=2,
    max_timeout=2000,
    verbose=True,
    show_progress=True,
    language=language,
    invalidate_cache=False,
    do_not_cache=False,
    fast_mode=True, # fast_mode=True doesn't work with result_type="markdown"
    ignore_errors=True,
    split_by_page=True,
    disable_ocr=True,
    target_pages=target_pages  # for testing purposes use target_pages="0,80" to only parse the first and last page 
    )

    file_extractor = {".pdf": pdf_parser}

    documents = SimpleDirectoryReader(input_files=pdf_document, 
                                      file_extractor=file_extractor,
                                      filename_as_id=True,
                                      required_exts=[".pdf"],
                                      num_files_limit=1).load_data()
    
    return documents


def transform(documents: List[Document]) -> List[Document]:
    transformed_documents = []
    for document in documents:
        transformed_documents.append(
            Document(
                text=document.text,
                metadata=document.metadata,
                excluded_llm_metadata_keys=["file_name", "file_path", "file_type", "file_size", "creation_date", "last_modified_date", "total_pages_in_original_pdf", "size_of_original_pdf(MB)"],
                excluded_embed_metadata_keys = ["file_path", "file_type", "file_size", "creation_date", "last_modified_date", "total_pages_in_original_pdf", "size_of_original_pdf(MB)"],
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
        )
    return transformed_documents

def load(documents: List[Document]):
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    qdrant_client = QdrantClient(url=config["QDRANT_ENDPOINT"], 
                             api_key=config["QDRANT_API_KEY"])
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="pillpal_documents")
    
    pipeline = IngestionPipeline(
        name="pillpal_ingestion_pipeline",
        project_name="pillpal_bot",
        transformations=[text_splitter, embedding_model],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding_model)

    return nodes, index


if __name__ == "__main__":
    documents = extract()
    documents = add_metadata_to_documents(documents)
    documents = transform(documents)
    nodes, index = load(documents)