
from scrapy.linkextractors import LinkExtractor 
import scrapy #Version 2.4.1
import requests
import os

class ConcordiaSiteSpider(scrapy.Spider):
    
    name = "ConcordiaWebSiteCrawler"
    upperBound = 0
    start_urls = ["https://www.concordia.ca/"]
    allowed_domains = ["concordia.ca"]
    directoryName = os.getcwd() + "\\HTML\\"
    page_links = set()
    
    def __init__(self,file_count,*args,**kwargs):
        self.upperBound = int(file_count)
        super(ConcordiaSiteSpider,self).__init__(**kwargs)
          
    def process_value(self,value):
        url = str(value)
        
        if (not str(value).startswith("http")):
            url = self.start_urls[0]+str(value)
        
        try:
            web_page = requests.get(url)
            if (web_page.status_code == 200 and url!="https://www.concordia.ca/maps/parking.html"):
                if( len(self.page_links) >= self.upperBound):
                    raise scrapy.exceptions.CloseSpider()
                else:
                    self.page_links.add(value)
                    with open(self.directoryName+"HTML"+str(len(self.page_links))+".html","wb") as f:
                        f.write(web_page.content)
    
        except requests.exceptions.ConnectionError:
        
            return value

        return value
     
    
    def parse(self, response):
        try:
             self.page_links = LinkExtractor(canonicalize=True, unique=True, process_value=self.process_value).extract_links(response)
        except:
            print("Catched Exception :"+str(len(self.page_links)))
  


           
            
       
            
            
            
      
        
       


