from flask import Flask
from flask import request
from flask import jsonify
from flask import session
from flask_cors import CORS
# from flask import before_request
import requests
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path
import cassio
from llama_index import StorageContext, VectorStoreIndex
from llama_index.vector_stores import CassandraVectorStore
from llama_index import download_loader
import openai
from llama_index.llms import OpenAI
from functools import wraps
import os
from dotenv import load_dotenv

load_dotenv()

# MY_ENV_VAR = os.getenv('MY_ENV_VAR')

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key' 

ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = 'default_keyspace'

cassio.init(
    database_id=ASTRA_DB_ID,
    token=ASTRA_DB_APPLICATION_TOKEN,
    keyspace=ASTRA_DB_KEYSPACE,
)

global_vector_retriever = None

def after_this_endpoint(func_to_run_after):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Run the original function
            response = func(*args, **kwargs)
            # Run the additional function
            func_to_run_after()
            # TODO: Make function call
            # Return the original response
            return response
        return wrapper
    return decorator

def load_pdf():
    global global_vector_retriever
    # openai.api_key = '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'
    cassandra_store = CassandraVectorStore(table="nasa", embedding_dimension=1536)
    # session["cassandra_store"]=cassandra_store

    PDFReader = download_loader("PDFReader")
    # session["PDFReader"]=PDFReader


    loader = PDFReader()
    # session["loader"] = loader
    documents = loader.load_data(file=Path('output3_Final.pdf'))

    storage_context = StorageContext.from_defaults(vector_store=cassandra_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(similarity_top_k=200)
    global_vector_retriever=retriever
    

    # question = "tell me deatil about COMMIT ID: 21"
    # answer = "RISD"


    # contexts = retriever.retrieve(question)
    # context_list = [n.get_content() for n in contexts]
    
    # prompt = "\n\n".join(context_list + [question])
    # llm = OpenAI(model="gpt-4-0125-preview")
    # response = llm.complete(prompt)
    # llm = Gemini(model="models/gemini-pro")
    # response = llm.complete(prompt)
    # print(str(response))


# Formatting functions
def format_file_details(commit_details):
    if not commit_details:
        return "no file changes were recorded in this commit."
    
    detail_sentences = []
    for detail in commit_details:
        sentence = f"the '{detail['file name']}' file was {detail['status']}, encompassing {detail['file changes']} change(s) in total, which included {detail['file additions']} addition(s) and {detail['file deletions']} deletion(s)."
        detail_sentences.append(sentence)
    return " and ".join(detail_sentences) + "."

def format_commit_comments(commit_comments):
    if not commit_comments:
        return "there were no comments on this commit."
    
    comment_sentences = []
    for comment in commit_comments:
        sentence = f"{comment['commenter']} remarked, '{comment['comment']}'"
        comment_sentences.append(sentence)
    return " and ".join(comment_sentences) + "."

def format_commit(commit, commit_id):
    file_details_str = format_file_details(commit.get("commit details", []))
    commit_comments_str = format_commit_comments(commit.get("commit comments", []))

    commit_paragraph_format = """ 
    Commit ID: {id}
    In a recent update to the repository, Author: {Author[Name]}, reachable at {Author[Email]}, made a commit on {Author[Date]}. This commit, identified by the SHA hash '{sha}', addressed an issue described in the commit message as "{Message}". The detailed record of this commit can be viewed online at {URL of Current Commit}, and its API endpoint is accessible via GitHub API at {url}.

    The commit involved a total of {stats[total]} change(s), breaking down into {stats[additions]} addition(s) and {stats[deletions]} deletion(s). Specifically, {file_details}.

    Furthermore, this commit attracted comments from other contributors, notably {commit_comments}.
    """

    return commit_paragraph_format.format(
        id=commit_id,
        # Author=commit["Author"],
        file_details=file_details_str,
        commit_comments=commit_comments_str,
        **commit  # Unpacking other keys like 'sha', 'Message', etc.

    )

def generate_pdf(commits, filename="output3_Final.pdf"):
    formatted_paragraphs = [format_commit(commit, commit_id) for commit_id, commit in enumerate(commits, 1)]
    wrapped_paragraphs = ['\n'.join(textwrap.wrap(paragraph, 100)) for paragraph in formatted_paragraphs]
    formatted_text = "\n\n".join(wrapped_paragraphs)

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    def add_text_to_page(text, canvas):
        textobject = canvas.beginText(40, height - 40)
        textobject.setFont("Helvetica", 10)
        textobject.setLeading(14)

        for line in text.split('\n'):
            if textobject.getY() < 40:
                canvas.drawText(textobject)
                canvas.showPage()
                textobject = canvas.beginText(40, height - 40)
                textobject.setFont("Helvetica", 10)
                textobject.setLeading(14)
            textobject.textLine(line)

        canvas.drawText(textobject)

    add_text_to_page(formatted_text, c)
    c.save()


@app.before_request
def load_session():
    my_secret = os.getenv("SECRET_MY")


    headers = {
        'Authorization': f'token {my_secret}'
    }
    session["headers"]=headers


@app.route('/fetch_commits', methods=['POST'])
@after_this_endpoint(load_pdf)
def fetch_commits():
    data = request.json
    username = data['username']
    openai_key = data['openai_key']
    openai.api_key=openai_key
    repo_name = data['repo_name']

    url = f'https://api.github.com/repos/{username}/{repo_name}/commits'
    page = 1
    all_commits = []

    while True:
        response = requests.get(url, params={'per_page': 100, 'page': page},headers=session["headers"])
        if response.status_code != 200:
            print(f"Error: Unable to fetch data, status code: {response.status_code}")
            break
        commits = response.json()
        if not commits:
            print("breaking")
            break

        print(f"page {page}")
        cno=0
        for commit in commits:
            cno+=1
            print(f"    cno {cno}")
            author_info = commit["commit"]["author"]
            
            # 'sha': commit['sha'],
            
            parent=commit["parents"][0]["html_url"]  if len(commit["parents"])>0 else ""
            filtered_commit = {
                "Author": {
                    "Name": author_info.get("name", ""),
                    "Email": author_info.get("email", ""),
                    "Date": author_info.get("date", "")
                },
                'sha': commit['sha'],
                "Message": commit["commit"].get("message", ""),
                "URL of Current Commit": commit.get("html_url", ""),
                "url": commit.get("url", ""),
                "Previous Commit": parent
            }
            # Commiting list 
            url_commit=commit.get("url", False)
            if url_commit:
                response = requests.get(url_commit,headers=session["headers"])
                response=response.json()
                # print(response)
                filtered_commit["stats"] = response["stats"]
                comments_url = response.get("comments_url",[])
                files=response["files"]
                file_changes=[]
                for file in files:
                    file_dict={
                        "file name":file["filename"],
                        "file changes":file["changes"],
                        "file additions":file["additions"],
                        "file deletions":file["deletions"],
                        "status":file["status"],
                        "raw url":file["raw_url"]
                    }
                    file_changes.append(file_dict)
                
                filtered_commit["commit details"]= file_changes

                if comments_url:
                    responses = requests.get(comments_url,headers=session["headers"])
                    
                    responses = responses.json()
                    # print(responses)
                    comment_list=[]
                    # TODO: add time and creator status
                    for response in responses:
                        comment_dict={
                            "comment":response.get("body",""),
                            "commenter":response.get("user",{"login":""})["login"]
                        }
                        comment_list.append(comment_dict)
                    filtered_commit["commit comments"]=comment_list
                    
            
                    # 'files': [file['filename'] for file in commit['files']]
            
            all_commits.append(filtered_commit)

        page += 1

    generate_pdf(all_commits)

    return jsonify({"status": "success", "message": "PDF generated successfully"})


@app.route('/ask', methods=['POST'])
def qna():
    global global_vector_retriever
    if global_vector_retriever:
        data = request.json
        question=data['question']
        contexts=global_vector_retriever.retrieve(question)
        
        context_list = [n.get_content() for n in contexts]
        
        prompt = "\n\n".join(context_list + [question])
        llm = OpenAI(model="gpt-4-0125-preview")
        response = llm.complete(prompt)
        print(str(response))
        return jsonify({"answer": str(response)}), 200 
    else:
        load_pdf()
        return jsonify({"error": "Vector retriever not initialized"}), 404
if __name__ == '__main__':
    app.run(port=80)
