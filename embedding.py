from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import re


def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '_', name).strip().replace(' ', '_')

def createVectorStore(file_path, file_name):
  loader = PyPDFLoader(file_path)
  documents = loader.load()
  vector_store_dir = f'.\\faiss-stores\\{file_name}\\'
  #Split the Document into chunks for embedding and vector storage.
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
  all_splits = text_splitter.split_documents(documents)

  embeddings = CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY'), model="embed-multilingual-v3.0")

  vector = FAISS.from_documents(all_splits, embeddings)
  os.makedirs(vector_store_dir, exist_ok=True)

  vector.save_local(folder_path=vector_store_dir)


if __name__ == '__main__':
  directory_path = "./rag-content/becoming-imaginal/"
  os.environ['USER_AGENT'] = 'myagent'
  load_dotenv('./.env')
 
  for filename in os.listdir(directory_path):
      if filename.endswith(".pdf"):
          file_path = os.path.join(directory_path, filename)
          file_name = sanitize_filename(os.path.splitext(filename)[0])  # Extract filename without extension
          print(file_name)
          print("start with {}".format(file_name))
          createVectorStore(file_path, file_name)
          print("finished with {}".format(file_name))