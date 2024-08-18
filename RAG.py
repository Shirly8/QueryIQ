import os
import shutil
import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma



CHROMA_PATH = "./VectorDB"
DATA_PATH = './Files'
db = None


# Function processes all PDFs
def load_documents():
    os.chdir('/Users/shirleyhuang/Documents/Apps/RAG')
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

#Split text into chunks of 800 texts
def split_documents(documents: list[Document]):
    chunk_size = 800
    chunk_overlap = int(chunk_size * 0.1) #10% overlap

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex= False,
    )
    return text_splitter.split_documents(documents)

#Create page IDs (Files/Aretti.pdf: page number, chunk number)
def createIds(chunks):
    last_page_id = None
    index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        page_id = f"{source}:{page}"

        #Incremenet ID of the chunk if pageID on lastpage
        if page_id == last_page_id:
            index +=1
        else:
            index = 0
        
        #Assign the chunkID
        id = f"{page_id}:{index}"
        last_page_id = page_id

        #Add it as meta-data
        chunk.metadata["id"] = id

    return chunks


