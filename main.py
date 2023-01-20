# Import necessary libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import argparse
import string
import nltk.corpus
from nltk.util import ngrams
import joblib
#import gensim
import re
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import tempfile
from io import StringIO
from PIL import  Image
import pytesseract
#from pytesseract import*
#import cv2
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
import spacy_streamlit
#import en_core_web_sm
from collections import Counter
from nltk.tokenize import sent_tokenize
import docx2txt
import pdfplumber
import requests
from bs4 import BeautifulSoup

#1. Translator Imports
from mtranslate import translate
import os
from gtts import gTTS
import base64
import datetime as dt
import googletrans
from googletrans import Translator
from langdetect import detect

#2. Youtube Transcript Extractor
# This is a python API which allows you to get the transcript/subtitles for a given YouTube video. 
from youtube_transcript_api import YouTubeTranscriptApi
# components.iframe takes a URL as its input. This is used for situations where you want to include 
# an entire page within a Streamlit app.
# Load a remote URL in an iframe.
import streamlit.components.v1 as components


# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Import the custom modules 
import Text_Summarisation as ts
from annotated_text import annotated_text
# https://pypi.org/project/spacy/

# import sys
# sys.argv=['']
# del sys

# Sidebar options
option = st.sidebar.selectbox('Navigation', 
["Overview",
 "Multilingual Text to Speech Translator",
 "Text Summarization",
 "Text Annotation"])

st.set_option('deprecation.showfileUploaderEncoding', False)

#*****************************************************************************************

if option == 'Overview':

    #st.subheader("Text Analysis")
    # Title of the application 
    st.title('Text Analysis')
    
    image = Image.open('./images/TextAnalysis.png')
    st.image(image, caption='**Text Analysis - Language Translation, Information Extraction, Text Summarization, Text Annotation**')

    
    st.write("Text mining (also known as text analysis), is the process of transforming unstructured text into structured data for easy analysis. Text mining uses natural language processing (NLP), allowing machines to understand the human language and process it automatically.")
    st.write("Text mining is an automatic process that uses natural language processing to extract valuable insights from unstructured text. By transforming data into information that machines can understand, text mining automates the process of classifying texts by sentiment, topic, and intent.")
    st.write("My application has tools for data translation, data summarization and data annotation in it. The user interface is an important aspect of my application. The user interface offers multiple tools to choose from using a left dropdown option.")        
    st.subheader("1. Multilingual Text-to-Speech Translator")
    st.write("Language Translation is the process of conveying a source language text clearly and completely in a target language. Translation allows information to be transferred across languages, expanding accessibility of the information.")
    st.write("For spreading new ideas, information and knowledge, the language translation is necessary. To effectively communicate between different cultures, language translation is important. The objective of this Python project is to translate a piece of text into another language.")
    st.write("How to use this tool")
    st.write("I. Choose Multilingual Text to Speech Translator option from the drop down")
    st.write("II. choose option to type/upload files or paste the youtube URL link") # upload image file
    st.write("III. Select the language from the language list")
    st.write("IV. Click on Translate button")
    st.write("V. Once file is processed it will display the translated text in the translated text window and also display the audio file you can listen or download it.")
    
    st.subheader("2. Text Summarization")
    st.write("Text summarization refers to the technique of shortening long pieces of text. The intention is to create a coherent and fluent summary having only the main points outlined in the document. Automatic text summarization methods are greatly needed to address the ever-growing amount of text data available online to both better help discover relevant information and to consume relevant information faster. Machine learning models are usually trained to understand documents and distill the useful information before outputting the required summarized texts.")
    st.write("How to use this tool")
    st.write("I. Choose Text Summarization option from the drop down")
    st.write("II. choose option to type or upload files or web scrapping or paste the youtube URL link")
    st.write("III. Once file is processed, We can view the content, EDA/VDA and Select the techinques from the list using which it will display the summary of text.")

    st.subheader("3. Text Annotation")   
    st.write("Text annotation in machine learning (ML) is the process of assigning labels to a digital file or document and its content. This is an NLP method where different types of sentence structures are highlighted by various criteria.")
    st.write("How to use this tool")
    st.write("I. Choose Text Annotation option from the drop down")
    st.write("II. choose option to Type/Enter the sentence or upload files or paste the youtube URL link")
    st.write("III. choose option NER to see the named entities of Text data OR Choose Option of Tokenization to see the Parts of Speech of text.")
    
#*****************************************************************************************

# 2. Multilingual Translator

elif option == 'Multilingual Text to Speech Translator':

   
    # read language dataset
    df = pd.read_excel(os.path.join('data', 'language.xlsx'),sheet_name='wiki')
    df.dropna(inplace=True)
    lang = df['name'].to_list()
    langlist=tuple(lang)
    langcode = df['iso'].to_list()
    
    # create dictionary of language and 2 letter langcode
    lang_array = {lang[i]: langcode[i] for i in range(len(langcode))}
    
    st.image("./images/Translator.gif")
    
        
    inputtext =""
    options = st.sidebar.radio("Please choose one option",('Type or Write', 'Youtube URL')) #'Image'
    
    if options == 'Type or Write':
        inputtext = st.text_area("INPUT",height=200)

        docx_file = st.file_uploader("OR Choose a file", type=["pdf","docx","txt"])
        if docx_file is not None:
        
            file_details = {"filename":docx_file.name, "filetype":docx_file.type,
                                    "filesize":docx_file.size}
            if docx_file.type == "text/plain":
                # Read as string (decode bytes to string)
                inputtext = str(docx_file.read(),encoding = "ISO-8859-1")
    
                #st.write(inputtext)
            elif docx_file.type == "application/pdf":
                inputtext =""
                try:
                    with pdfplumber.open(docx_file) as pdf:
                        for page in pdf.pages:
                            inputtext = inputtext + "\n" + page.extract_text()
                            #st.write(inputtext)
                except:
                    st.write("None")
                
            else:
                inputtext = docx2txt.process(docx_file)   
    
        
    if options == "Youtube URL":
        st.write("YouTube Transcript Extractor")
        URL = st.sidebar.text_input("Paste YouTube URL:","https://www.youtube.com/watch?v=i2jwZcWicSY")
        
        if "=" in URL:
            ID = URL.split("=")[1] # splitting the URL based on = sign id is the value present after = sign
            # An embedded video lets you borrow the video from another platform. 
            # Visitors can watch the video on your website without leaving the current page. 
            embedurl = "https://www.youtube.com/embed/" + ID  
            if "&" in embedurl: embedurl = embedurl.split("&")[0]
        try:
            # components.iframe takes a URL as its input. This is used for situations where you want to include 
            # an entire page within a Streamlit app.
            # Load a remote URL in an iframe.
            # The iFrame component lets you load external URL elements (including other web pages) 
            # in your project within an iframe.
            # The parent site can define aspects of the iframe such as size, position and security context.
            # Here I can alter the video’s width, height, frame size, and other elements for 
            # optimal viewing on my landing pages. 
            components.iframe(embedurl, width=500, height=250)
        except:
            st.error("YouTube URL Required")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(ID)
            l = [t['text'] for t in transcript]
            description = " ".join(l)
        except:
            pass
        
        try :
            if len(description) > 0:
                inputtext = st.text_area("Extracted Transcript",description,height=200)
                st.download_button(label="Download Transcript",data=description,file_name=str(ID) + ".txt",mime="text/plain")
        except Exception as e:
                st.error("Transcript Not available for this video")
        
        
    # if options == "Image":
        
    #     def load_image(image_file):
    #         img = Image.open(image_file)
    #         return img    
        
    #     image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    #     ap = argparse.ArgumentParser(description='Fooo')
    #     args = ap.parse_args()
    #     ap.add_argument("-l", "--lang", required=True,help="language that Tesseract will use when OCR'ing")
    #     ap.add_argument("-t", "--to", type=str, default="en",help="language that we'll be translating to")
    #     ap.add_argument("-p", "--psm", type=int, default=13,help="Tesseract PSM mode")
    #     args = vars(ap.parse_args())
   
    #     if image_file is not None:
    #           # To View Uploaded Image
    #           st.image(load_image(image_file),width=250)
    #           image = cv2.imread(args["image"])
    #           rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    #           #tessdata_dir_config = r'--tessdata-dir "/app/.apt/usr/share/tesseract-ocr/tessdata"'
    #           pytesseract.pytesseract.tesseract_cmd = r'/app/.apt/usr/bin/tesseract'
    #           #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sandesh Singh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    #           options = "-l {} --psm {}".format(args["lang"], args["psm"]) # Image.open(image_file)
    #           inputtext = pytesseract.image_to_string(rgb,lang='eng+hin+mar+pan+guj+ind+kor+urd+tam+telfra+ara+asm+jpn+kan', config=options) # eng+hin+mar+pan+guj+ind+kor+urd+tam+telfra+ara+asm+jpn+kan
    #           if st.button("Display Extracted Text"):
    #               st.text(inputtext[:-1]) 

    if st.button("Detect Input Language And Play Audio File"):                  
            lang_detect = detect(inputtext) 
            st.write("Language of Input Text is :", lang_detect)
            # Passing the text and language to the engine, 
            # here we have marked slow=False. Which tells 
            # the module that the converted audio should 
            # have a high speed
            input_audio = gTTS(text=inputtext, lang=lang_detect, slow=False)
            # Saving the converted audio in a mp3 file named
            # welcome 
            input_audio.save("input_audio.mp3")
            # Playing the converted file
            #os.system("mpg321 input_audio.mp3")
            input_audio_read = open('input_audio.mp3', 'rb')
            input_audio_bytes = input_audio_read.read()
            bin_str = base64.b64encode(input_audio_bytes).decode()
            st.audio(input_audio_bytes, format='audio/mp3')
	
	
    option = st.selectbox('Select Language',langlist)
    st.sidebar.write("1. Languages are pulled from language.xlsx. If translation is available it will be displayed in Translated Text window.")
    st.sidebar.write("2. In addition if text-to-Speech is supported it will display audio file to play and download." )
        
    speech_langs = {
        "af": "Afrikaans",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "bs": "Bosnian",
        "ca": "Catalan",
        "cs": "Czech",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "gu": "Gujarati",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "id": "Indonesian",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jw": "Javanese",
        "km": "Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "la": "Latin",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mr": "Marathi",
        "my": "Myanmar (Burmese)",
        "ne": "Nepali",
        "nl": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Filipino",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh-CN": "Chinese"
    }
    
	
    # function to decode audio file for download
    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    
    c1,c2 = st.columns([4,3])
    
    if st.button("Translate"):
        
        # I/O
        if len(inputtext) > 0 :
            try:
                #translator=Translator()
                #output = translator.translate(inputtext, dest=lang_array[option]).text
                output = translate(inputtext,lang_array[option])
                with c1:
                    st.text_area("TRANSLATED TEXT",output,height=200)
                # if speech support is available will render autio file
                if option in speech_langs.values():
                    with c2:
                        aud_file = gTTS(text=output, lang=lang_array[option], slow=False)
                        aud_file.save("lang.mp3")
                        audio_file_read = open('lang.mp3', 'rb')
                        audio_bytes = audio_file_read.read()
                        bin_str = base64.b64encode(audio_bytes).decode()
                        st.audio(audio_bytes, format='audio/mp3')
                        st.markdown(get_binary_file_downloader_html("lang.mp3", 'Audio File'), unsafe_allow_html=True)
            except Exception as e:
                st.error(e)


#*****************************************************************************************

#3. Text Summarizer

if option == 'Text Summarization':
    
    st.write("**Text Summarization**")
    st.image("./images/Summarization.jpg")
    st.write("Text summarization refers to the technique of shortening long pieces of text. The intention is to create a coherent and fluent summary having only the main points outlined in the document.")
    
    def generate_ngrams(words_list, n):
        ngrams_list = []
     
        for num in range(0, len(words_list)):
            ngram = ' '.join(words_list[num:num + n])
            ngrams_list.append(ngram)     
        return ngrams_list
    
    
    strAllTexts =""
    options = st.sidebar.radio("Please choose one option",('Upload File', 'Web Scrapping','Type or Write', 'Youtube URL'))
    if options == 'Upload File':
        docx_file = st.sidebar.file_uploader("Choose a file", type=["pdf","docx","txt"])
        if docx_file is not None:
            
            file_details = {"filename":docx_file.name, "filetype":docx_file.type,
                                    "filesize":docx_file.size}
            if docx_file.type == "text/plain":
                # Read as string (decode bytes to string)
                strAllTexts = str(docx_file.read(),encoding = "ISO-8859-1")
    
                #st.write(strAllTexts)
            elif docx_file.type == "application/pdf":
                strAllTexts =""
                try:
                    with pdfplumber.open(docx_file) as pdf:
                        for page in pdf.pages:
                            strAllTexts = strAllTexts + "\n" + page.extract_text()
                            #st.write(strAllTexts)
                except:
                    st.write("None")
                
            else:
                strAllTexts = docx2txt.process(docx_file)
               
    
    if options == 'Type or Write':
            strAllTexts=st.text_area(label="Please enter text", max_chars=35000, height=400)
    if options == 'Web Scrapping':
            linkURL = st.text_input(label="Please enter URL of website", max_chars=100, help="From the entered URL we will fetch all the content which will be present inside <p> tag only")
            try:
                r=requests.get(linkURL)
                bs= BeautifulSoup(r.text,'html.parser')

                p=bs.find_all('p')
            
                for lines in p:
                    strAllTexts = strAllTexts + "\n"  + str(lines.text)
            except:
                st.write("Please enter valid URL")

    if options == "Youtube URL":
        st.write("YouTube Transcript Extractor")
        URL = st.sidebar.text_input("Paste YouTube URL:","https://www.youtube.com/watch?v=Y8Tko2YC5hA")    #https://www.youtube.com/watch?v=ukzFI9rgwfU
        
        if "=" in URL:
            ID = URL.split("=")[1]
            embedurl = "https://www.youtube.com/embed/" + ID
            if "&" in embedurl: embedurl = embedurl.split("&")[0]
        try:
            components.iframe(embedurl, width=500, height=250)
        except:
            st.error("YouTube URL Required")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(ID)
            l = [t['text'] for t in transcript]
            description = " ".join(l)
        except:
            pass
        
        try :
            if len(description) > 0:
                strAllTexts = st.text_area("Extracted Transcript",description,height=200)
                st.download_button(label="Download Transcript",data=description,file_name=str(ID) + ".txt",mime="text/plain")
        except Exception as e:
                st.error("Transcript Not available for this video")     
		
    if len(strAllTexts):
        genre = st.sidebar.radio("Please choose one option",('View File Content', 'EDA / VDA', 'Summary'))
    


        #storing number of char
        vTotChars = len(strAllTexts)
        
        #Storing number of lines
        lstAllLines = strAllTexts.split('\r')
        vTotLines = len(lstAllLines)
        #print(lstAllLines)
        
        
        lstTmpWords = []
        for i in range(0,len(lstAllLines)):
            strLine = lstAllLines[i]
            lstWords = strLine.split(" ")
            lstTmpWords.append(lstWords)
        
        # split each line into a list of words
        lstTmpWords = []
        for i in range(0,len(lstAllLines)):
            strLine = lstAllLines[i]
            lstWords = strLine.split(" ")
            lstTmpWords.append(lstWords)

            # merge in single list
        lstAllWords = []    
        for lstWords in lstTmpWords:
            for strWord in lstWords:
                lstAllWords.append(strWord)
    
        vTotWords = len(lstAllWords)
        


        #st.text(nLineSmry)
            
        #############################################################
        # compute word freq & word weight
        #############################################################
        from nltk.tokenize import word_tokenize
        lstAllWords = word_tokenize(strAllTexts)

        # Convert the tokens into lowercase: lower_tokens
        lstAllWords = [t.lower() for t in lstAllWords]
        
        # retain alphabetic words: alpha_only
        
        lstAllWords = [t.translate(str.maketrans('','','01234567890')) for t in lstAllWords]
        lstAllWords = [t.translate(str.maketrans('','',string.punctuation)) for t in lstAllWords]
        
        # remove all stop words
        # original found at http://en.wikipedia.org/wiki/Stop_words
        lstStopWords = nltk.corpus.stopwords.words('english')
        lstAllWords = [t for t in lstAllWords if t not in lstStopWords]
        
        # remove all bad words ...
        # original found at http://en.wiktionary.org/wiki/Category:English_swear_words
        lstBadWords = ["arse","ass","asshole","bastard","bitch","bloody","bollocks","child-fucker",
                       "cunt","damn","fuck","goddamn","godsdamn","hell","motherfucker","shit",
                       "shitass","whore"]
        lstAllWords = [t for t in lstAllWords if t not in lstBadWords]
        
        # remove application specific words
        lstSpecWords = ['rt','via','http','https','mailto']
        lstAllWords = [t for t in lstAllWords if t not in lstSpecWords]
        
        # retain words with len > 3
        lstAllWords = [t for t in lstAllWords if len(t)>3]
        
        # import WordNetLemmatizer
        # https://en.wikipedia.org/wiki/Stemming
        # https://en.wikipedia.org/wiki/Lemmatisation
        # https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
        from nltk.stem import WordNetLemmatizer
        # instantiate the WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        # Lemmatize all tokens into a new list: lemmatized
        lstAllWords = [wordnet_lemmatizer.lemmatize(t) for t in lstAllWords]
            
        text = ' '.join(lstAllWords)
        
        # create a Counter with the lowercase tokens: bag of words - word freq count
        # import Counter
        from collections import Counter
        dctWordCount = Counter(lstAllWords)
        
        
        #print('\n*** Convert To Dataframe ***')
        dfunigramscolle  = pd.DataFrame.from_dict(dctWordCount, orient='index').reset_index()
        dfunigramscolle.columns = ['Word','Freq'] 
        dfunigramscolle = dfunigramscolle.sort_values(by='Freq',ascending=False)
        dfunigramscolle=dfunigramscolle.head(10)
        
        #Biagram

        biagram = generate_ngrams(lstAllWords, 2)
        
        biagramcolle = Counter(biagram)        
                     
        dfbiagramcolle  = pd.DataFrame.from_dict(biagramcolle, orient='index').reset_index()
        dfbiagramcolle.columns = ['Word','Freq']
        dfbiagramcolle = dfbiagramcolle.sort_values(by='Freq',ascending=False)
        dfbiagramcolle = dfbiagramcolle.head(10)

        #Trigram

        trigram = generate_ngrams(lstAllWords, 3)
        
        trigramcolle = Counter(trigram)        
                     
        dftrigramcolle  = pd.DataFrame.from_dict(trigramcolle, orient='index').reset_index()
        dftrigramcolle.columns = ['Word','Freq']
        dftrigramcolle = dftrigramcolle.sort_values(by='Freq',ascending=False)
        dftrigramcolle = dftrigramcolle.head(10)
        
        if genre == 'View File Content':
        #if st.button('View Content'):
            st.markdown("**File content**")
            st.markdown(strAllTexts)

        
        if genre == 'EDA / VDA' :
            
            col1, col2, col3, col4 = st.columns(4)
            if options == 'Upload File':
                col1.metric("**File Size(MB)**", round(docx_file.size / (1024 * 1024), 4))
            col2.metric("**Total Words**", vTotWords)
            col3.metric("**Total Characters**", vTotChars)
            if options == 'Upload File':
                if docx_file.type == "text/plain":
                    col4.metric("**Total Lines**", vTotLines)
            
            # Sidebar options
            option = st.sidebar.selectbox('Navigation',
            ["Unigram - 1-Ngram",
             "Biagram - 2-Ngram",
             "Trigram - 3-Ngram",
             "N-Gram"])
            
            st.set_option('deprecation.showfileUploaderEncoding', False)
            
            if option == 'Unigram - 1-Ngram':
                #with col11:
                st.subheader("Top keywords")
                st.dataframe(dfunigramscolle)
                st.subheader("Unigram")
                st.subheader("Word Freq Count - Top 10")
                fig = plt.figure(figsize = (10, 6))
                ax = sns.barplot(x='Word', y='Freq', data = dfunigramscolle)
                #ax = plt.bar(dfunigramscolle['Word'], dfunigramscolle['Freq'])
                plt.xlabel("Words")
                plt.ylabel("Freq")   
                plt.xticks(rotation=90)
                st.pyplot(fig)

            if option == 'Biagram - 2-Ngram':
                #with col11:
                st.subheader("Top keywords")
                st.dataframe(dfbiagramcolle)
                st.subheader("Bigram")
                st.subheader("Word Freq Count - Top 10")
                fig = plt.figure(figsize = (10, 6))
                ax = sns.barplot(x='Word', y='Freq', data = dfbiagramcolle)
                #ax = plt.bar(dfbiagramcolle['Word'], dfbiagramcolle['Freq'])
                plt.xlabel("Words")
                plt.ylabel("Freq")  
                plt.xticks(rotation=90)
                st.pyplot(fig)
                
            if option == 'Trigram - 3-Ngram':
                #with col11:
                st.subheader("Top keywords")
                st.dataframe(dftrigramcolle)
                st.subheader("Trigram")
                st.subheader("Word Freq Count - Top 10")
                fig = plt.figure(figsize = (10, 6))
                ax = sns.barplot(x='Word', y='Freq', data = dfbiagramcolle)
                #ax = plt.bar(dftrigramcolle['Word'], dftrigramcolle['Freq'])
                plt.xlabel("Words")
                plt.ylabel("Freq")  
                plt.xticks(rotation=90)
                st.pyplot(fig)

            if option == 'N-Gram':
                # Function to plot the ngrams based on n and top k value

                def plot_ngrams(text, n=4, topk=15):
                    '''
                    Function to plot the most commonly occuring n-grams in bar plots 
                    
                    ARGS
                    	text: data to be enterred
                    	n: n-gram parameters
                    	topk: the top k phrases to be displayed
                
                    '''
                
                    st.write('Creating N-Gram Plot..')
                
                    #text = lstAllWords
                    tokens = text.split()
                    
                    # get the ngrams 
                    ngram_phrases = ngrams(tokens, n)
                    
                    # Get the most common ones 
                    most_common = Counter(ngram_phrases).most_common(topk)
                    
                    # Make word and count lists 
                    words, counts = [], []
                    for phrase, count in most_common:
                        word = ' '.join(phrase)
                        words.append(word)
                        counts.append(count)
                    df = pd.DataFrame()
                    df['Word']=words
                    df['Freq']=counts
                    # Plot the barplot 
                    title = "Most Common " + str(n) + "-grams in the text"
                    plt.title(title)
                    fig = plt.figure(figsize=(10, 6))
                    ax = sns.barplot(x='Word', y='Freq', data = df)
                    #ax = plt.bar(words, counts)
                    plt.xlabel("n-grams found in the text")
                    plt.ylabel("Ngram frequencies")
                    plt.xticks(rotation=90)
                    #st.pyplot(fig)
                    plt.show()

                    
                n = st.sidebar.slider("N for the N-gram", min_value=4, max_value=8, step=1, value=4)
                topk = st.sidebar.slider("Top k most common phrases", min_value=10, max_value=50, step=5, value=10)

            	# Add a button 
                if st.button("Generate N-Gram Plot"): 
              		# Plot the ngrams
                      plot_ngrams(text, n=n, topk=topk)
                      st.pyplot()
          
                
            #with col12:
            st.subheader("Word Cloud for top words")
            #plot word cloud
            # word cloud options
            # https://www.datacamp.com/community/tutorials/wordcloud-python
            #print('\n*** Plot Word Cloud - Top 100 ***')
            import matplotlib.pyplot as plt
            from wordcloud import WordCloud
            
            dftWordCount  = pd.DataFrame.from_dict(dctWordCount, orient='index').reset_index()
            dftWordCount.columns = ['Word','Freq'] 
            dftWordCount = dftWordCount.sort_values(by='Freq',ascending=False)
            dftWordCount=dftWordCount.head(30)
            
            d = {}
            for a, x in dftWordCount[['Word','Freq']].values:
                d[a] = x 
            wordcloud = WordCloud(background_color="white")
            wordcloud.generate_from_frequencies(frequencies=d)
            fig = plt.figure(figsize = (5, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(fig)

            
        if genre == 'Summary' :
            
            model = st.sidebar.selectbox("Model Select", ["Latent Semantic Analysis", "TextRank", "LexRank"]) # "GenSim"
            #ratio = st.sidebar.slider("Select summary ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            
            if model == "Latent Semantic Analysis":
            
                nLineSmry = st.number_input("Please enter number of line for summary you would like to see", min_value=1)
                                
                
                # word weight = word-count / max(word-count)
                # replace word count with word weight
                max_freq = sum(dctWordCount.values())
                for word in dctWordCount.keys():
                    dctWordCount[word] = (dctWordCount[word]/max_freq)
                # weights of words
                
                #############################################################
                # create sentences / lines
                #############################################################
                
                # split scene_one into sentences: sentences
                from nltk.tokenize import sent_tokenize
                lstAllSents = sent_tokenize(strAllTexts)
    
                
                # convert into lowercase
                lstAllSents = [t.lower() for t in lstAllSents]
                
                # remove punctuations
    
                lstAllSents = [t.translate(str.maketrans('','','[]{}<>')) for t in lstAllSents]
                lstAllSents = [t.translate(str.maketrans('','','0123456789')) for t in lstAllSents]
                
                # sent score
                dctSentScore = {}
                for Sent in lstAllSents:
                    for Word in nltk.word_tokenize(Sent):
                        if Word in dctWordCount.keys():
                            if len(Sent.split(' ')) < 30:
                                if Sent not in dctSentScore.keys():
                                    dctSentScore[Sent] = dctWordCount[Word]
                                else:
                                    dctSentScore[Sent] += dctWordCount[Word]
                
                
                #############################################################
                # summary of the article
                #############################################################
                # The "dctSentScore" dictionary consists of the sentences along with their scores. 
                # Now, top N sentences can be used to form the summary of the article.
                # Here the heapq library has been used to pick the top 5 sentences to summarize the article
                import heapq
                lstBestSents = heapq.nlargest(nLineSmry, dctSentScore, key=dctSentScore.get)
                # for vBestSent in lstBestSents:
                #     st.write(vBestSent)
                
                # # final summary
                # strTextSmmry = '. '.join(lstBestSents) 
                # strTextSmmry = strTextSmmry.translate(str.maketrans(' ',' ','\n'))
                # st.write(strTextSmmry)
                                    
                
                #############################################################
                # lsa summary
                # https://iq.opengenus.org/latent-semantic-analysis-for-text-summarization/
                #############################################################
                
                #import sumy
                ##We're choosing a plaintext parser here, other parsers available for HTML etc.
                from sumy.parsers.plaintext import PlaintextParser
                from sumy.nlp.tokenizers import Tokenizer
                ##We're choosing Luhn, other algorithms are also built in
                from sumy.summarizers.lsa import LsaSummarizer as Summarizer
                from sumy.nlp.stemmers import Stemmer
                from sumy.utils import get_stop_words
                
                # parser object with AllTexts
                parserObject = PlaintextParser.from_string(strAllTexts,Tokenizer("english"))
                stemmer = Stemmer("english")
                summarizer = Summarizer(stemmer)
    
    
                # summarizer object
                
                st.write("**********Summary of your document**********")
                summarizer.stop_words = get_stop_words("english")
    
                summaryList = []
                ##Summarize the document with 5 sentences
                my_summary = summarizer(parserObject.document, nLineSmry) # sentence count set to 10
                summaryList = list(my_summary)
                for i in summaryList:
                    st.write(str(i).strip())
                    
            text = strAllTexts       
            # if model == "GenSim":
            #    	sentence_count = len(sent_tokenize(text))
            #    	st.write("Number of sentences:", sentence_count)
            #    	ratio = st.sidebar.slider("Select summary ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            #     out = ts.text_sum_gensim(text, ratio=ratio)
            #     st.write("**Summary Output:**", out)
            #     st.write("Number of output sentences:", len(sent_tokenize(out)))
             			# st.write(out)
            if model == "TextRank":
                sentence_count = len(sent_tokenize(text))
               	st.write("Number of sentences:", sentence_count)
               	ratio = st.sidebar.slider("Select summary ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
                out = ts.text_sum_text(text, ratio=ratio)
                st.write("**Summary Output:**", out)
                st.write("Number of output sentences:", len(sent_tokenize(out)))
              			# st.write(out)
            if model == "LexRank":
                sentence_count = len(sent_tokenize(text))
               	st.write("Number of sentences:", sentence_count)
               	ratio = st.sidebar.slider("Select summary ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
                out = ts.text_sum_lex(text, ratio=ratio)
              			# st.write(out)
                st.write("**Summary Output:**", out)
                st.write("Number of output sentences:", len(sent_tokenize(out)))   
                

if option == "Text Annotation":
    
    st.write("**Text Annotation :**")
    st.write("Text annotation in machine learning (ML) is the process of assigning labels to a digital file or document and its content. This is an NLP method where different types of sentence structures are highlighted by various criteria.")
    st.image("./images/Text Annotation.gif")
    
    text_input = ""
	#st.header("Enter the text")
    options = st.sidebar.radio("Please choose one option",('Type/Enter the sentence', 'Upload File', 'Web Scrapping','Youtube URL'))
    
    if options == "Type/Enter the sentence":
    
        st.markdown("**Example Random Sentence:** Barack Hussein Obama II (born August 4, 1961) is an American attorney and politician who served as the 44th President of United States from January 20, 2009 to January 20, 2017. A member of the Democratic Party, he was the first African American to serve as president. He was previously a United States Senator from lllinois and a member of the lllinois State Senate.")
        
        text_input = st.text_area("Type/Enter the sentence")
    
    
    if options == "Upload File":
    
        uploaded_file = st.file_uploader("or Upload a file", type=["doc", "docx", "pdf", "txt"])
        if uploaded_file is not None:
            text_input = uploaded_file.getvalue()
        
    if options == 'Web Scrapping':
            linkURL = st.text_input(label="Please enter URL of website", max_chars=100, help="From the entered URL we will fetch all the content which will be present inside <p> tag only")
            try:
                r=requests.get(linkURL)
                bs= BeautifulSoup(r.text,'html.parser')

                p=bs.find_all('p')
            
                for lines in p:
                    text_input = text_input + "\n"  + str(lines.text)
            except:
                st.write("Please enter valid URL")
                
    if options == "Youtube URL":
        st.write("YouTube Transcript Extractor")
        URL = st.sidebar.text_input("Paste YouTube URL:","https://www.youtube.com/watch?v=Y8Tko2YC5hA")    #https://www.youtube.com/watch?v=ukzFI9rgwfU
        
        if "=" in URL:
            ID = URL.split("=")[1]
            embedurl = "https://www.youtube.com/embed/" + ID
            if "&" in embedurl: embedurl = embedurl.split("&")[0]
        try:
            components.iframe(embedurl, width=500, height=250)
        except:
            st.error("YouTube URL Required")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(ID)
            l = [t['text'] for t in transcript]
            description = " ".join(l)
        except:
            pass
        
        try :
            if len(description) > 0:
                text_input = st.text_area("Extracted Transcript",description,height=200)
                st.download_button(label="Download Transcript",data=description,file_name=str(ID) + ".txt",mime="text/plain")
        except Exception as e:
                st.error("Transcript Not available for this video")   
    
    nlp = spacy.load("en_core_web_sm")
    ner = spacy.load("en_core_web_sm")
    doc = ner(str(text_input))
    
    options = st.sidebar.radio("Please choose one option",('NER', 'Tokenization'))
    
    if options == "NER":
        # Display 
        spacy_streamlit.visualize_ner(doc, labels=ner.get_pipe('ner').labels)
        st.image("./images/NER_2.png")
        
    if options == "Tokenization":
        spacy_streamlit.visualize_tokens(doc, attrs=['text', 'pos_'])   
        st.image("./images/POS_tag.png")

#*****************************************************************************************


