from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import os
from dotenv import load_dotenv


load_dotenv('./.env')

parent_dir = "faiss-stores/"
dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
embeddings =  CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY'), model="embed-multilingual-v3.0" )
print(dirs)
indexes = []
for d in dirs:
    path = os.path.join(parent_dir, d)
    index = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    indexes.append(index)

base_index = indexes[0]
for idx in indexes[1:]:
    base_index.merge_from(idx)

base_index.save_local("faiss-stores/merged_index")