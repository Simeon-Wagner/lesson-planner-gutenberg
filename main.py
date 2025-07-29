
from flask import Flask, request, jsonify
from flask import render_template

# Imports for RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

app = Flask(__name__)

@app.route('/')
def index():

    levels = [
        {"value": "1. Curso", "label": "1. Curso", "disabled": False},
        {"value": "2. Curso", "label": "2. Curso", "disabled": False},
        {"value": "3. Curso", "label": "3. Curso", "disabled": False}
        # {"value": "1. Curso"},
        # {"value": "2. Curso"},
        # {"value": "3. Curso"}
    ]

    subjects = [
        {"value": "Matematica", "label": "Matematica", "disabled": False},
        {"value": "Castellano", "label": "Castellano", "disabled": False},
        {"value": "Guarani", "label": "Guarani", "disabled": False}
    ]
    return render_template('index.html', levels=levels, subjects=subjects)

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
        with open("C:\\Users\\wagne\\Desktop\\gutenberg\\ai-gutenberg\\lesson-planner-gutenberg\\prompt_templates\\test.txt", "r") as file:
            prompt_template_string = file.read()

        template_vars = {
            "input": data["topic"],
            "subject" :  data["subject"], #this is important for the retriever chain. Do not rename.
            "level" : data["level"]
        }
        print(template_vars)

        # prompt_template_string = prompt_template_string.replace("[topic]",template_vars["topic"] )
        # prompt_template_string = prompt_template_string.replace("[level]",template_vars["level"] )
        # prompt_template_string = prompt_template_string.replace("[subject]",template_vars["subject"] )

   
        store_guide = "faiss-stores/becoming-imaginal-2-feb-2025-es"
        store_curriculum = "faiss-stores/bachillerato-cientifico-con-enfasis-en-ciencia-sociales"

        # BUILD THE RETRIEVAL CHAIN
        llm = ChatCohere(cohere_api_key=os.getenv('COHERE_API_KEY'), model="command-r-08-2024")
        embeddings =  CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY'), model="embed-multilingual-v3.0" )
        index_guide = FAISS.load_local(store_guide, embeddings, allow_dangerous_deserialization=True)
        index_curriculum = FAISS.load_local(store_curriculum, embeddings, allow_dangerous_deserialization=True)
        index_guide.merge_from(index_curriculum)

        prompt = ChatPromptTemplate.from_template(prompt_template_string)

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = index_guide.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke(template_vars)

        # PREPARE THE OUTPUT
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
    app.run()