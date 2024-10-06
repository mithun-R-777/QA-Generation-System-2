import streamlit as st
import wikipedia
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs
from haystack.nodes import TfidfRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from main import print_qa, QuestionGenerator

def main():
    # Set the Streamlit app title
    st.title("QA Generation System ")

    # Select the input type
    inputs = ["Input Paragraph", "Wikipedia Examples"]
    input_type = st.selectbox("Select an input type:", inputs)

    # Initialize wiki_text as an empty string
    wiki_text = ""

    # Handle different input types
    if input_type == "Input Paragraph":
        # Allow user to input text paragraph
        wiki_text = st.text_area("Input paragraph:", height=200)

    elif input_type == "Wikipedia Examples":
        # Define topics for selection
        topics = ["Deep Learning", "Machine Learning"]
        selected_topic = st.selectbox("Select a topic:", topics)

        # Retrieve Wikipedia content based on the selected topic
        if selected_topic:
            wiki = wikipedia.page(selected_topic)
            wiki_text = wiki.content

        # Display the retrieved Wikipedia content (optional)
        st.text_area("Retrieved Wikipedia content:", wiki_text, height=200)

    # Preprocess the input text
    wiki_text = clean_wiki_text(wiki_text)

    # Allow user to specify the number of questions to generate
    num_questions = st.slider("Number of questions to generate:", min_value=1, max_value=20, value=5)

    model_name = "deepset/roberta-base-squad2"

    # Button to generate questions
    if st.button("Generate Questions"):
        document_store = InMemoryDocumentStore()

        # Convert the preprocessed text into a document
        document = {"content": wiki_text}
        document_store.write_documents([document])

        # Initialize a TfidfRetriever
        retriever = TfidfRetriever(document_store=document_store)

        # Initialize a FARMReader with the selected model
        reader = FARMReader(model_name_or_path=model_name, use_gpu=False)

        # Initialize the question generation pipeline
        pipe = ExtractiveQAPipeline(reader, retriever)

        # Initialize the QuestionGenerator
        qg = QuestionGenerator()

        # Generate multiple-choice questions
        qa_list = qg.generate(
            wiki_text,
            num_questions=num_questions,
            answer_style='multiple_choice'
        )

        # Display the generated questions and answers
        st.header("Generated Questions and Answers:")
        for idx, qa in enumerate(qa_list):
            # Display the question
            st.write(f"Question {idx + 1}: {qa['question']}")

            # Display the answer options
            if 'answer' in qa:
                for i, option in enumerate(qa['answer']):
                    correct_marker = "(correct)" if option["correct"] else ""
                    st.write(f"Option {i + 1}: {option['answer']} {correct_marker}")

            # Add a separator after each question-answer pair
            st.write("-" * 40)







# Run the Streamlit app
if __name__ == "__main__":
    main()



# import streamlit as st
# import wikipedia
# from haystack.document_stores import InMemoryDocumentStore
# from haystack.utils import clean_wiki_text, convert_files_to_docs
# from haystack.nodes import TfidfRetriever, FARMReader
# from haystack.pipelines import ExtractiveQAPipeline
# from main import print_qa, QuestionGenerator
# import torch

# def main():
#     # Set the Streamlit app title
#     st.title("Question Generation using Haystack and Streamlit")

#     # Select the input type
#     inputs = ["Input Paragraph", "Wikipedia Examples"]
#     input_type = st.selectbox("Select an input type:", inputs, key="input_type")

#     # Initialize wiki_text as an empty string (to avoid UnboundLocalError)
#     wiki_text = """ Deep learning is the subset of machine learning methods based on artificial neural networks (ANNs) with representation learning. The adjective "deep" refers to the use of multiple layers in the network. Methods used can be either supervised, semi-supervised or unsupervised.Deep-learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.Artificial neural networks were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, artificial neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog. ANNs are generally seen as low quality models for brain function."""

#     # Handle different input types
#     if input_type == "Input Paragraph":
#         # Allow user to input text paragraph
#         wiki_text = st.text_area("Input paragraph:", height=200, key="input_paragraph")

#     elif input_type == "Wikipedia Examples":
#         # Define options for selecting the topic
#         topics = ["Deep Learning", "Machine Learning"]
#         selected_topic = st.selectbox("Select a topic:", topics, key="wiki_topic")

#         # Retrieve Wikipedia content based on the selected topic
#         if selected_topic:
#             wiki = wikipedia.page(selected_topic)
#             wiki_text = wiki.content

#         # Display the retrieved Wikipedia content (optional)
#         st.text_area("Retrieved Wikipedia content:", wiki_text, height=200, key="wiki_text")

#     # Allow user to specify the number of questions to generate
#     num_questions = st.slider("Number of questions to generate:", min_value=1, max_value=20, value=5, key="num_questions")

#     # Allow user to specify the model to use
#     model_options = ["deepset/roberta-base-squad2", "deepset/roberta-base-squad2-distilled", "bert-large-uncased-whole-word-masking-squad2", "deepset/flan-t5-xl-squad2"]
#     model_name = st.selectbox("Select model:", model_options, key="model_name")

#     # Button to generate questions
#     if st.button("Generate Questions", key="generate_button"):
#         # Initialize the document store
#         with open('wiki_txt.txt', 'w', encoding='utf-8') as f:
#             f.write(wiki_text)
#         document_store = InMemoryDocumentStore()
#         doc_dir = "/content"
#         docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
#         document_store.write_documents(docs)
#         retriever = TfidfRetriever(document_store=document_store)

#         # # Convert the input text paragraph or Wikipedia content into a document
#         # document = {"content": wiki_text}
#         # document_store.write_documents([document])

#         # Initialize a TfidfRetriever
#         # retriever = TfidfRetriever(document_store=document_store)

#         # Initialize a FARMReader with the selected model
#         reader = FARMReader(model_name_or_path=model_name, use_gpu=False)

#         # Initialize the question generation pipeline
#         pipe = ExtractiveQAPipeline(reader, retriever)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Initialize the QuestionGenerator
#         qg = QuestionGenerator()

#         # Generate multiple-choice questions
#         qa_list = qg.generate(wiki_text, num_questions=num_questions, answer_style='multiple_choice')

#         # Display the generated questions and answers
#         st.header("Generated Questions and Answers:")
#         for idx, qa in enumerate(qa_list):
#             # Display the question
#             st.write(f"Question {idx + 1}: {qa['question']}")

#             # Display the answer options
#             if 'answer' in qa:
#                 for i, option in enumerate(qa['answer']):
#                     correct_marker = "(correct)" if option["correct"] else ""
#                     st.write(f"Option {i + 1}: {option['answer']} {correct_marker}")

#             # Add a separator after each question-answer pair
#             st.write("-" * 40)

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()



























# # import streamlit as st
# # import wikipedia
# # from haystack.document_stores import InMemoryDocumentStore
# # from haystack.utils import clean_wiki_text, convert_files_to_docs
# # from haystack.nodes import TfidfRetriever, FARMReader
# # from haystack.pipelines import ExtractiveQAPipeline
# # from main import print_qa, QuestionGenerator

# # def main():
# #     # Set the Streamlit app title
# #     st.title("Question Generation using Haystack and Streamlit")
# #     # select the input type 
# #     inputs = ["Input Paragraph", "Wikipedia Examples"]
# #     input=st.selectbox("Select a Input Type :", inputs)
# #     if(input=="Input Paragraph"):
# #         # Allow user to input text paragraph
# #         wiki_text = st.text_area("Input paragraph:", height=200)

# #         # # Allow user to specify the number of questions to generate
# #         # num_questions = st.slider("Number of questions to generate:", min_value=1, max_value=20, value=5)

# #         # # Allow user to specify the model to use
# #         # model_options = ["deepset/roberta-base-squad2", "deepset/roberta-base-squad2-distilled", "bert-large-uncased-whole-word-masking-squad2","deepset/flan-t5-xl-squad2"]
# #         # model_name = st.selectbox("Select model:", model_options)

# #         # # Button to generate questions
# #         # if st.button("Generate Questions"):
# #         #     qno=0
          
# #         #     # Initialize the document store
# #         #     document_store = InMemoryDocumentStore()
        
# #         #     # Convert the input text paragraph into a document
# #         #     document = {"content": wiki_text}
# #         #     document_store.write_documents([document])
        
# #         #     # Initialize a TfidfRetriever
# #         #     retriever = TfidfRetriever(document_store=document_store)
        
# #         #     # Initialize a FARMReader with the selected model
# #         #     reader = FARMReader(model_name_or_path=model_name, use_gpu=False)
        
# #         #     # Initialize the question generation pipeline
# #         #     pipe = ExtractiveQAPipeline(reader, retriever)
        
# #         #     # Initialize the QuestionGenerator
# #         #     qg = QuestionGenerator()
        
# #         #     # Generate multiple-choice questions
# #         #     qa_list = qg.generate(
# #         #     wiki_text, 
# #         #     num_questions=num_questions, 
# #         #     answer_style='multiple_choice')
# #         #     print("QA List Structure:")
# #         #     # Display the generated questions and answers
# #         #     st.header("Generated Questions and Answers:")
# #         #     for qa in qa_list:
# #         #         opno=0
                
# #         #         # Display the question
# #         #         st.write(f"Question: {qno+1}{qa['question']}")

# #         #     # Display the answer options
# #         #         if 'answer' in qa:
# #         #             for idx, option in enumerate(qa['answer']):
# #         #             # Indicate if the option is correct
# #         #                 correct_marker = "(correct)" if option["correct"] else ""
# #         #                 st.write(f"Option {idx + 1}: {option['answer']} {correct_marker}")
        
# #         #     # Add a separator after each question-answer pair
# #         #         st.write("-" * 40)
    
# #     if(input == "Wikipedia Examples"):   
# #         # Define options for selecting the topic
# #         topics = ["Deep Learning", "MachineLearning"]
# #         selected_topic = st.selectbox("Select a topic:", topics)

# #         # Retrieve Wikipedia content based on the selected topic
# #         if selected_topic:
# #             wiki = wikipedia.page(selected_topic)
# #             wiki_text = wiki.content

# #         # Display the retrieved Wikipedia content in a text area (optional)
# #         st.text_area("Retrieved Wikipedia content:", wiki_text, height=200)

# #         # # Allow user to specify the number of questions to generate
# #         # num_questions = st.slider("Number of questions to generate:", min_value=1, max_value=20, value=5)

# #         # # Allow user to specify the model to use
# #         # model_options = ["deepset/roberta-base-squad2", "deepset/roberta-base-squad2-distilled", "bert-large-uncased-whole-word-masking-squad2","deepset/flan-t5-xl-squad2"]
# #         # model_name = st.selectbox("Select model:", model_options)

# #         # # Button to generate questions
# #         # if st.button("Generate Questions"):
# #         #     # Initialize the document store
# #         #     document_store = InMemoryDocumentStore()
        
# #         #     # Convert the retrieved Wikipedia content into a document
# #         #     document = {"content": wiki_text}
# #         #     document_store.write_documents([document])
        
# #         #     # Initialize a TfidfRetriever
# #         #     retriever = TfidfRetriever(document_store=document_store)
        
# #         #     # Initialize a FARMReader with the selected model
# #         #     reader = FARMReader(model_name_or_path=model_name, use_gpu=False)
        
# #         #     # Initialize the ExtractiveQAPipeline
# #         #     pipeline = ExtractiveQAPipeline(reader, retriever)
        
# #         #     # Initialize the QuestionGenerator
# #         #     qg = QuestionGenerator()
        
# #         #     # Generate multiple-choice questions
# #         #     qa_list = qg.generate(
# #         #         wiki_text, 
# #         #         num_questions=num_questions, 
# #         #         answer_style='multiple_choice'
# #         #     )
        
# #         #     # Display the generated questions and answers
# #         #     st.header("Generated Questions and Answers:")
# #         #     for idx, qa in enumerate(qa_list):
# #         #         # Display the question
# #         #         st.write(f"Question {idx + 1}: {qa['question']}")

# #         #         # Display the answer options
# #         #         if 'answer' in qa:
# #         #             for i, option in enumerate(qa['answer']):
# #         #                 correct_marker = "(correct)" if option["correct"] else ""
# #         #                 st.write(f"Option {i + 1}: {option['answer']} {correct_marker}")
            
# #         #         # Add a separator after each question-answer pair
# #         #         st.write("-" * 40)
    
# #     # Allow user to specify the number of questions to generate
# #     num_questions = st.slider("Number of questions to generate:", min_value=1, max_value=20, value=5)
# #     # Allow user to specify the model to use
# #     model_options = ["deepset/roberta-base-squad2", "deepset/roberta-base-squad2-distilled", "bert-large-uncased-whole-word-masking-squad2","deepset/flan-t5-xl-squad2"]
# #     model_name = st.selectbox("Select model:", model_options)

# #     # Button to generate questions
# #     if st.button("Generate Questions"):
# #         qno=0
          
# #         # Initialize the document store
# #         document_store = InMemoryDocumentStore()
        
# #         # Convert the input text paragraph into a document
# #         document = {"content": wiki_text}
# #         document_store.write_documents([document])
        
# #         # Initialize a TfidfRetriever
# #         retriever = TfidfRetriever(document_store=document_store)
        
# #         # Initialize a FARMReader with the selected model
# #         reader = FARMReader(model_name_or_path=model_name, use_gpu=False)
        
# #         # Initialize the question generation pipeline
# #         pipe = ExtractiveQAPipeline(reader, retriever)
        
# #         # Initialize the QuestionGenerator
# #         qg = QuestionGenerator()
        
# #         # Generate multiple-choice questions
# #         qa_list = qg.generate(
# #             wiki_text, 
# #             num_questions=num_questions, 
# #             answer_style='multiple_choice')
# #         print("QA List Structure:")
# #         # Display the generated questions and answers
# #         st.header("Generated Questions and Answers:")
# #         for qa in qa_list:
# #             opno=0
                
# #             # Display the question
# #             st.write(f"Question: {qno+1}{qa['question']}")

# #         # Display the answer options
# #             if 'answer' in qa:
# #                 for idx, option in enumerate(qa['answer']):
# #                 # Indicate if the option is correct
# #                     correct_marker = "(correct)" if option["correct"] else ""
# #                     st.write(f"Option {idx + 1}: {option['answer']} {correct_marker}")
        
# #         # Add a separator after each question-answer pair
# #             st.write("-" * 40)

# # # Run the Streamlit app
# # if __name__ == "__main__":
# #     main()




# # # import streamlit as st
# # # import re
# # # import pke
# # # import contractions
# # # import wikipedia
# # # import logging
# # # from haystack.document_stores import InMemoryDocumentStore
# # # from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
# # # from transformers.pipelines import question_answering
# # # from haystack.nodes import TfidfRetriever
# # # from haystack.pipelines import ExtractiveQAPipeline
# # # from haystack.nodes import FARMReader
# # # import torch

# # # from main import print_qa
# # # from main import QuestionGenerator

# # # def main():
# # #     # Initialize Streamlit app
# # #     st.title("Question Generation using Haystack and Streamlit")

# # #     # Allow user to input text paragraph
# # #     wiki_text = st.text_area("Input paragraph:", height=200)

# # #     # Allow user to specify the number of questions to generate
# # #     num_questions = st.slider("Number of questions to generate:", min_value=1, max_value=20, value=5)

# # #     # Allow user to specify the model to use
# # #     model_options = ["deepset/roberta-base-squad2", "deepset/roberta-base-squad2-distilled", "bert-large-uncased-whole-word-masking-squad2"]
# # #     model_name = st.selectbox("Select model:", model_options)

# # #     # Button to generate questions
# # #     if st.button("Generate Questions"):
# # #         # Initialize the document store
# # #         document_store = InMemoryDocumentStore()
        
# # #         # Convert the input text paragraph into a document
# # #         document = {"content": wiki_text}
# # #         document_store.write_documents([document])
        
# # #         # Initialize a TfidfRetriever
# # #         retriever = TfidfRetriever(document_store=document_store)
        
# # #         # Initialize a FARMReader with the selected model
# # #         reader = FARMReader(model_name_or_path=model_name, use_gpu=False)
        
# # #         # Initialize the question generation pipeline
# # #         pipe = ExtractiveQAPipeline(reader, retriever)
        
# # #         # Initialize the QuestionGenerator
# # #         qg = QuestionGenerator()
        
# # #         # Generate multiple-choice questions
# # #         qa_list = qg.generate(
# # #         wiki_text, 
# # #         num_questions=num_questions, 
# # #         answer_style='multiple_choice')
# # #         print("QA List Structure:")
# # #         # Display the generated questions and answers
# # #         st.header("Generated Questions and Answers:")
# # #         for qa in qa_list:
# # #             # Display the question
# # #             st.write(f"Question: {qa['question']}")

# # #             # Display the answer options
# # #             if 'answer' in qa:
# # #                 for idx, option in enumerate(qa['answer']):
# # #                     # Indicate if the option is correct
# # #                     correct_marker = "(correct)" if option["correct"] else ""
# # #                     st.write(f"Option {idx + 1}: {option['answer']} {correct_marker}")
        
# # #             # Add a separator after each question-answer pair
# # #             st.write("-" * 40)
# # #         # for qa in qa_list:
# # #         #     print(qa)
        
# # #         # # Proceed with displaying the generated questions
# # #         # st.header("Generated Questions:")
# # #         # for qa in qa_list:
# # #         #     st.write(f"Question: {qa['question']}")
# # #         #     # Adjust the code to match the structure of the output
# # #         #     if 'answers' in qa:
# # #         #         for idx, answer in enumerate(qa['answers']):
# # #         #             prefix = f"Option {idx + 1}:"
# # #         #             if answer["correct"]:
# # #         #                 prefix += " (correct)"
# # #         #             st.write(f"{prefix} {answer['text']}")
# # #         #     else:
# # #         #         st.write("No answers available for this question.")
# # #         #     st.write("")  # Add an empty line between each question for better readability

# # # # Run the Streamlit app
# # # if __name__ == "__main__":
# # #     main()
