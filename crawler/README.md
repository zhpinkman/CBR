# Crawler

* `final_data` directory contains the final data that we used for our analysis. It contains different columns with all the extracted representations such as `explanations`, `structure`, `goals`, and the `counterarguments` of the arguments. 

* `sentence_reprsentation_extraction` contains the code for extracting different case representations such as `explanations`, `goals`, `counterarguments`, and `structure` of the arguments. As it uses the OpenAI api and needs the api key, you have to include all the keys in the `config.py` file as a dictionary with the keys being indices starting from 0 and values being the keys.