
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS



# Imports for RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings

# IO Imports
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():

    levels = [
        {"value": "1. Curso", "label": "1. Curso", "disabled": False},
        {"value": "2. Curso", "label": "2. Curso", "disabled": False},
        {"value": "3. Curso", "label": "3. Curso", "disabled": False}
    ]

    subjects = [
        {"value": "Matematica", "label": "Matematica", "disabled": False},
        {"value": "Castellano", "label": "Castellano", "disabled": False},
        {"value": "Guarani", "label": "Guarani", "disabled": False}
    ]

    periods = [
        {"value": "35", "label": "1h", "disabled": False},
        {"value": "70", "label": "2h", "disabled": False},
        {"value": "105", "label": "3h", "disabled": False}
    ]
    return render_template('index.html', levels=levels, subjects=subjects, periods=periods)

@app.route('/faq/')
def faq():
    return render_template('faq.html')
    

@app.route('/generateLessonPlan', methods=["POST"])
def getLessonPlan():
    try:
        data = request.get_json()
        print("Received data:", data)

        # PREPARE THE PROMPT
        # Here they used the mode parameter to specify wether a lesson is regenerated or generated the first time. 
        with open("prompt_templates/test.txt", "r", encoding='utf-8') as file:
            prompt_template_string = file.read()

        template_vars = {
            "input": data["topic"], #this is important for the retriever chain. Do not rename.
            "subject" :  data["subject"],
            "periods" : int(data["periods"])/35,
            "period_length" : data["periods"],
            "level" : data["level"],
            "learningObjectives" : data["learning_obj"]
        }
        print(template_vars)

        merged_vector_store = "faiss-stores/merged_index"

        # BUILD THE RETRIEVAL CHAIN
        llm = ChatCohere(cohere_api_key=os.getenv('COHERE_API_KEY'), model="command-r-08-2024")
        embeddings =  CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY'), model="embed-multilingual-v3.0" )
        index_guide = FAISS.load_local(merged_vector_store, embeddings, allow_dangerous_deserialization=True)

        prompt = ChatPromptTemplate.from_template(prompt_template_string)

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = index_guide.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke(template_vars)

        pages = []

        for p in response["context"]:
            pages.append(p.metadata["page"])

        return jsonify({
            "answer": response["answer"],
            "pages":list(set(pages))
            })
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 400


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    load_dotenv('./.env')
    app.debug = True
    app.run(host='0.0.0.0', port=8080)