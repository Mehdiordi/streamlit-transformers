from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st
import wikipedia

wikipedia = WikipediaAPIWrapper()

st.set_page_config(page_title="Streamlit App", page_icon="static/res/favicon.png")

st.markdown(
    '''
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
    </style>
    <a href="https://pythonpythonme.netlify.app/index.html">
    <div class="center-image">
    <!--
    <img src="https://pythonpythonme.netlify.app/PythonPythonME.png" alt="Header image">
    -->
    </div>
    </a>
    <p></p>
    <p></p>
    <body>
        <header>
            <div>
                <h1>Streamlit Question Answering App</h1>
                <div class="center-image">
                <h1>🦜 🦚</h1>
                </div>
            </div>
        </header>
    </body>
    ''',
    unsafe_allow_html=True
)

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# User input
question_input = st.text_input("Question:")

if question_input:
    keywords = question_input.split()
    # Fetch context information using the Wikipedia toolkit based on keywords
    wikipedia = WikipediaAPIWrapper()
    context_input = wikipedia.run(' '.join(keywords))

    QA_input = {
        'question': question_input,
        'context': context_input
    }

    res = nlp(QA_input)

    # Display the answer
    st.text_area("Answer:", res['answer'])
    st.write("Score:", res['score'])

st.markdown(
    '''
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
        .follow-me {
            text-align: center;
        }
        .social-icons {
            display: flex;
            justify-content: center;
            list-style: none;
            padding: 0;
        }
        .social-icons li {
            margin: 0 10px;
        }
    </style>
    <body>
        <div class="center-image">
            <h4>Mehdi Ordikhani</h4>
        </div>
        <div class="center-image">
            <h4>Follow Me</h4>
        </div>
        <div class="center-image">
            <ul class="social-icons">
                <li><a href="https://www.linkedin.com/in/mehdi-ordikhani-seyedlar-11349b103/"><img src="https://img.freepik.com/premium-vector/linkedin-logo_578229-227.jpg" width="55" height="55" alt="LinkedIn"></a></li>
                <li><a href="https://github.com/mehdiordi"><img src="https://cdn.pixabay.com/photo/2022/01/30/13/33/github-6980894_1280.png" width="55" height="55" alt="GitHub"></a></li>
            </ul>
        </div>
        <footer class="footer">
            <div class="container">
                <div class="row">
                    <div class="center-image">
                    <!--
                        <p class="text-muted">© 2023-2024 PythonPythonME.</p>
                        <p>All rights reserved.</p>
                    -->
                    </div>
                </div>
            </div>
        </footer>
    </body>
    ''',
    unsafe_allow_html=True
)
