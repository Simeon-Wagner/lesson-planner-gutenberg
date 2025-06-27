
from flask import Flask, request, jsonify
from flask import render_template

# Imports for RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings

import config

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/faq/')
def faq():
    return render_template('faq.html')

@app.route('/generateLessonPlan', methods=["POST"])
def getLessonPlan():
    '''
        This endpoint generates a lesson plan based on the input provided in the POST request.
    
        The method accepts a JSON payload containing various parameters such as:
        - `mode`: Specifies the template to use for generating the lesson plan (either generate or regenerate).
        
        # Variables for "generate" and "regenerate"
        - `topic`: The main topic of the lesson.
        - `learningObjectives`: The learning objectives for the lesson.
        - `classGrade`: The class grade.
        - `studentLevel`: The level of the students (e.g., grade or proficiency).
        - `subject`: The subject of the lesson.

        # Variables only for "regenerate"
        - `focusMoreOn`: Areas to emphasize in the lesson.
        - `focusLessOn`: Areas to de-emphasize in the lesson.
        - `lp`: The previous lesson plan to adapt.
        - `specialNeeds`: A flag to include considerations for special needs.
        - `difficulty`: Specifies the difficulty level of the lesson plan.

    
        The method performs the following steps:
        1. Reads the appropriate prompt template based on the `mode`.
        2. Dynamically updates the template with the provided input variables.
        3. Adjusts the template based on optional flags like `genderIssues`, `specialNeeds`, and `difficulty`.
        4. Loads the appropriate FAISS vector store for the specified subject.
        5. Builds a retrieval chain using LangChain components, including a language model and retriever.
        6. Executes the retrieval chain to generate a response.
        7. Extracts relevant pages and prepares the output in JSON format.
    
        Returns:
            A JSON response containing:
            - `answer`: The generated lesson plan.
            - `pages`: A list of pages referenced in the response.
            - `subject`: The subject of the lesson plan.
    
        In case of an error, it returns a JSON response with the error message and a 400 status code.
    '''
    try:
        data = request.get_json()
        mode = data["mode"]

        # PREPARE THE PROMPT
        with open("prompt_templates/{}.txt".format(mode), "r") as file:
            prompt_template_string = file.read()

        template_vars = {
            "input" :  data["topic"], #this is important for the retriever chain. Do not rename.
            "learningObjectives" : data["learningObjectives"],
            "classGrade": data["classGrade"],
            "focusMoreOn" : data["focusMoreOn"],
            "focusLessOn" : data["focusLessOn"],
            "lp" : data["lp"]
        }

        if (data["genderIssues"]):
            prompt_template_string = prompt_template_string + """
                Please include considerations regarding gender issues in the lesson development and activities. Please provide an example therefore.
            """

        if (data["specialNeeds"]):
            prompt_template_string = prompt_template_string + """
                Please include considerations regarding special needs in the lesson development and activities. Please provide an example therefore.
            """

        if (data["difficulty"] != "no_choice"):
            prompt_template_string = prompt_template_string + """
                Please adapt the difficulty level of the lesson plan you have given me. Make it a bit {}.
            """.format(data["difficulty"])

        # LOOK FOR THE RIGHT VECTOR STORE
        lp_student_level = data["studentLevel"]
        lp_subject = data["subject"]

        store = ""
        store = "faiss_stores/s1_{}_student_2020_prototype".format(lp_subject)

        # BUILD THE RETRIEVAL CHAIN
        llm = ChatCohere(cohere_api_key="something", model="command-r-08-2024")
        vector = FAISS.load_local(store, CohereEmbeddings(cohere_api_key="something", model="embed-english-v3.0"), allow_dangerous_deserialization=True)

        prompt = ChatPromptTemplate.from_template(prompt_template_string)

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke(template_vars)

        # PREPARE THE OUTPUT
        pages = []

        for p in response["context"]:
            pages.append(p.metadata["page"])

        return jsonify({
            "answer": response["answer"],
            "pages":list(set(pages)),
            "subject": lp_subject,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    
    app.debug = True
    app.run()