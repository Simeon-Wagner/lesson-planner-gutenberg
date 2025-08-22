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
        {"value": "1. Grado", "label": "1. Grado", "disabled": False},
        {"value": "2. Grado", "label": "2. Grado", "disabled": False},
        {"value": "3. Grado", "label": "3. Grado", "disabled": False},
        {"value": "4. Grado", "label": "4. Grado", "disabled": False},
        {"value": "5. Grado", "label": "5. Grado", "disabled": False},
        {"value": "6. Grado", "label": "6. Grado", "disabled": False},
        {"value": "7. Grado", "label": "7. Grado", "disabled": False},
        {"value": "8. Grado", "label": "8. Grado", "disabled": False},
        {"value": "9. Grado", "label": "9. Grado", "disabled": False},
        {"value": "1. Curso", "label": "1. Curso", "disabled": False},
        {"value": "2. Curso", "label": "2. Curso", "disabled": False},
        {"value": "3. Curso", "label": "3. Curso", "disabled": False}
    ]

subjects = [
    {"value": "Antropología Cultural", "label": "Antropología Cultural", "disabled": False},
    {"value": "Antropología Social", "label": "Antropología Social", "disabled": False},
    {"value": "Artes Plásticas", "label": "Artes Plásticas", "disabled": False},
    {"value": "Biología", "label": "Biología", "disabled": False},
    {"value": "Castellano", "label": "Castellano", "disabled": False},
    {"value": "Ciencias", "label": "Ciencias", "disabled": False},
    {"value": "Economia Financiera", "label": "Economia Financiera", "disabled": False},
    {"value": "Economia y Gestión", "label": "Economia y Gestión", "disabled": False},
    {"value": "Educacicon Vial", "label": "Educacicon Vial", "disabled": False},
    {"value": "Educación Física", "label": "Educación Física", "disabled": False},
    {"value": "Estadística", "label": "Estadística", "disabled": False},
    {"value": "Ética", "label": "Ética", "disabled": False},
    {"value": "Filosofía", "label": "Filosofía", "disabled": False},
    {"value": "Física", "label": "Física", "disabled": False},
    {"value": "Guaraní", "label": "Guaraní", "disabled": False},
    {"value": "Historia", "label": "Historia", "disabled": False},
    {"value": "Informática", "label": "Informática", "disabled": False},
    {"value": "Inglés", "label": "Inglés", "disabled": False},
    {"value": "Laboratorio Matemática", "label": "Laboratorio Matemática", "disabled": False},
    {"value": "Laboratorio de Ciencias", "label": "Laboratorio de Ciencias", "disabled": False},
    {"value": "Liderazgo", "label": "Liderazgo", "disabled": False},
    {"value": "Matemática", "label": "Matemática", "disabled": False},
    {"value": "Metodología", "label": "Metodología", "disabled": False},
    {"value": "Música", "label": "Música", "disabled": False},
    {"value": "Orientación Cristiana", "label": "Orientación Cristiana", "disabled": False},
    {"value": "Orientación Educacional", "label": "Orientación Educacional", "disabled": False},
    {"value": "Politica", "label": "Politica", "disabled": False},
    {"value": "Primeros Auxilios", "label": "Primeros Auxilios", "disabled": False},
    {"value": "Química", "label": "Química", "disabled": False},
    {"value": "Sicología", "label": "Sicología", "disabled": False},
    {"value": "Sociología", "label": "Sociología", "disabled": False},
    {"value": "Trab. Y Tecnología", "label": "Trab. Y Tecnología", "disabled": False},
    {"value": "Comunicación", "label": "Comunicación", "disabled": False},
    {"value": "Educación para la Salud", "label": "Educación para la Salud", "disabled": False},
    {"value": "Ciencias Sociales", "label": "Ciencias Sociales", "disabled": False},
    {"value": "Medio Natural", "label": "Medio Natural", "disabled": False},
    {"value": "Vida Social", "label": "Vida Social", "disabled": False},
    {"value": "Ciencias Naturales", "label": "Ciencias Naturales", "disabled": False},

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