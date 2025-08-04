
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS



# Imports for RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings

# OpenAI
from openai import OpenAI

# IO Imports
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

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

models = [
    {"value": "rag - cohere", "label": "rag - cohere", "disabled": False},
    {"value": "prompt only - openai", "label": "prompt only - openai", "disabled": False}
]

@app.route('/')
def index():
    return render_template('index.html', levels=levels, subjects=subjects, periods=periods, models=models)

@app.route('/faq/')
def faq():
    return render_template('faq.html')
    

def get_cohere_rag_response(formData: dict, prompt: str)-> (dict, int):
    merged_vector_store = "faiss-stores/merged_index"
    try:
        llm = ChatCohere(cohere_api_key=os.getenv('COHERE_API_KEY'), model="command-r-08-2024")
        embeddings =  CohereEmbeddings(cohere_api_key=os.getenv('COHERE_API_KEY'), model="embed-multilingual-v3.0" )
        index_guide = FAISS.load_local(merged_vector_store, embeddings, allow_dangerous_deserialization=True)

        prompt = ChatPromptTemplate.from_template(prompt)

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = index_guide.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke(formData)

        pages = []

        for p in response["context"]:
            pages.append(p.metadata["page"])

        return {"answer": response["answer"]}, 200
    
    except Exception as e:
            print(str(e))
            return {"error": str(e)}, 400
    

def get_openai_response(formData: dict, prompt: str) -> (dict, int):
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    with open("prompt_templates/system-prompt.txt" , "r", encoding="utf-8") as file:
        system_prompt = file.read()

    filled_prompt = prompt.format(**formData)

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": filled_prompt}
            ],
            temperature=0.7,
            max_tokens=5000  # adjust as needed
        )

        print(completion.choices[0].message.content)

        return  {'answer':completion.choices[0].message.content}, 200

    except Exception as e:
            print(str(e))
            return {"error": str(e)}, 400


@app.route('/generateLessonPlan', methods=["POST"])
def getLessonPlan():
    try:
            
        template_vars = {
            "input": "", #this is important for the retriever chain. Do not rename.
            "subject" :  "",
            "periods" : "",
            "period_length" : "",
            "model" : "",
            "level" : "",
            "learningObjectives" : "",
            "previousLessonPlan" : ""
        }

        template_vars["subject"] = request.form.get("subject")
        template_vars["level"] = request.form.get("level")
        template_vars["period_length"] =  request.form.get("periods")
        template_vars["periods"] =  int(template_vars["period_length"])/35
        template_vars["input"] = request.form.get("topic")
        template_vars["learningObjectives"] = request.form.get("learning_obj")
        template_vars["model"] = request.form.get("model")

        # Extract uploaded file
        print(template_vars)

        uploaded_file = request.files.get("previous_lesson_plan")
        if uploaded_file:
            template_vars["previousLessonPlan"] = uploaded_file.read().decode("utf-8")
            print("Loaded previous lesson data.")
        
        resp = 0
        lp  = {}

        # Check the model to use.
        if template_vars["model"] == "rag - cohere":
            with open("prompt_templates/rag-cohere-prompt.txt", "r", encoding='utf-8') as file:
                prompt_template_string = file.read()
            lp, resp = get_cohere_rag_response(formData=template_vars, prompt=prompt_template_string)

        elif template_vars["model"] == "prompt only - openai":
            with open("prompt_templates/openai-prompt-only.txt", "r", encoding='utf-8') as file:
                prompt_template_string = file.read()
            lp, resp = get_openai_response(formData=template_vars, prompt=prompt_template_string)


        if resp == 400:
            raise Exception("Something went wrong during generation")
        
        return jsonify(lp)
    
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