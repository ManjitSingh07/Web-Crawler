from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup #Version 4.9.3
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from nltk.stem import PorterStemmer
from afinn import Afinn



extracted_text_of_pages={}
tokens_of_pages={}



directory=os.getcwd() + "\\HTML\\"
            
        
    
def crawl_and_extract_data():
    
    page_number=1
    while(page_number<=len(os.listdir(directory))):
                             
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                with open(f, 'rb') as file:
                    html_file = file.read()
                    file.close()
                    soup = BeautifulSoup(html_file,'html.parser')
                    for script in soup("script"):
                        script.decompose()
                        
                    for style in soup("style"):
                        style.decompose()
                        
                    print("Extracting data from: " + filename)
                    final_string=re.sub(r'[^\w\s]',' ',soup.get_text().replace("\n"," "))
                    final_string=re.sub(r'[0-9]', '', final_string)
                    extracted_text_of_pages[filename]=final_string.lower()
                    page_number+=1
        
        
        

def tokenize_extracted_data():
    for filename in os.listdir(directory):
        tokenized_text = word_tokenize(extracted_text_of_pages[filename])
        tokens_of_pages[filename]=list(set(tokenized_text))
    


def remove_stop_words():
    stop_words = set(stopwords.words('english'))
    for filename in os.listdir(directory):
        for tokens in tokens_of_pages[filename]:
            if tokens in stop_words:
                tokens_of_pages[filename].remove(tokens)

def stemming():
    ps=PorterStemmer()
    for filename in os.listdir(directory):
        for tokens in tokens_of_pages[filename]:
            tokens=ps.stem(tokens)
    
def text_preprocessing():
    crawl_and_extract_data()
    tokenize_extracted_data()
    remove_stop_words()
    stemming()
    
def output_for_tf_idf():
    documents = tokens_of_pages.values()
    documents=list(documents)
    final_output=[]
    for doc in documents:
        final_output.append(" ".join(doc))
    return final_output
    
def compute_tf_idf_scores():
    documents=output_for_tf_idf()
    tfidfvectorizer = TfidfVectorizer(ngram_range = (1,1), sublinear_tf = True, min_df = 1, analyzer = 'word')
    train = tfidfvectorizer.fit_transform(documents)
    return train,tfidfvectorizer

def k_means_clustering():
    trained_data,tfidfvectorizer = compute_tf_idf_scores()
    print()
    print("Result with 6 clusters")
    
    kmeans = KMeans(n_clusters = 6)
    kmeans.fit(trained_data)
    
    print(kmeans.labels_)
    
    print()
    print('Topmost 50 terms in respective clusters are: ')
    print()
    
    Centroids = kmeans.cluster_centers_.argsort()[:,::-1]
    Top_50_Terms = tfidfvectorizer.get_feature_names()
    
    for i in range(6):
        print('Cluster number %d:' %i)
        print()
        print("Top 50 terms are:")
        print()
        print("[")
        result_terms = fetch_top_most_terms(i,Top_50_Terms,Centroids)
        print(result_terms)
        print("]")
        print()

    for j in range(0,6):
        result_terms = fetch_top_most_terms(j,Top_50_Terms,Centroids)
        afinn_score(j,result_terms)
        
    print()
    print("Result with 3 clusters")
    
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(trained_data)
    
    print(kmeans.labels_)
    
    print()
    print('Topmost 50 terms in respective clusters are: ')
    print()
    
    Centroids = kmeans.cluster_centers_.argsort()[:,::-1]
    Top_50_Terms = tfidfvectorizer.get_feature_names()
    
    for k in range(0,3):
        print('Cluster number %d:' %k)
        print()
        print("Top 50 terms are:")
        print()
        print("[")
        result_terms = fetch_top_most_terms(k,Top_50_Terms,Centroids)
        print(result_terms)
        print("]")
        print()

    for l in range(0,3):
        result_terms = fetch_top_most_terms(l,Top_50_Terms,Centroids)
        afinn_score(l,result_terms)
    print()
 
def afinn_score(cluster_number,centroids_term):
    afn = Afinn()
    sentiment = afn.score(centroids_term)
    print("Sentiment score of Cluster " +str(cluster_number)+" using AFINN: "+str(sentiment))
    
    
def fetch_top_most_terms(Cluster_number,Terms_list,centroids_list):
    top_most_terms = ','.join([Terms_list[Cluster_number] for Cluster_number in centroids_list[Cluster_number, :50]])
    return top_most_terms
    
    
    
def run():
    print()
    text_preprocessing()
    k_means_clustering()




if __name__ == "__main__":
    upper_bound = str(input("How many pages you want to crawl? "))
    os.system("scrapy crawl ConcordiaWebSiteCrawler -a file_count="+upper_bound)
    print()
    run()
    #os.remove(os.getcwd() + "\\HTML")

























