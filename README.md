# LDA-Feature-Based Sentiment: Amazon Customer Review

## Introduction
Amazon.com has started to get high demand in the retail category since the world was hit by the coronavirus pandemic crisis where many are increasingly relying on online purchases than ever before. Thus, understanding how consumers make decisions when shopping online has become an important subject to the e-commerce industry as consumers’ purchasing decisions will directly influence the sale of goods. This situation has made it a necessity for Amazon to obtain detailed information on consumers’ needs and formulate products that can meet their demands such as coffee products. 

## Objective
Therefore, the objective of this project is to extract product characteristics, choice patterns and consumer behaviour towards a product from the text by developing a topic modelling-based method, Latent Dirichlet Allocation (LDA). In addition, this project also developed Part of Speech tagging-feature model to improve the accuracy of the LDA model and optimize this model by finding the most relevant number of topics, K. 

## Methodology
This project applied a topic-sentiment modelling approach which is Latent Dirichlet Allocation (LDA) and VADER Sentiment Analysis to extract product characteristics and discover consumer preferences and behaviours over 16,275 of coffee products using customer review data and product metadata from Amazon.com which was extracted by Ni et al. (2019). 
![image](https://user-images.githubusercontent.com/58675575/178278677-72f38e9e-198b-4873-826a-2933b4c6201b.png)

## Results
The results of this study showed that Nouns, Verbs and Adverb are the best combination of word types, and the optimal number of K topics is 6. The characteristics of coffee extracted from this model are “Coffee Service”, “Coffee Quality”, “Price "," Coffee Bean "," Coffee Flavour "and" Packaging "and all these features are positive features because their positive sentiments have the highest number of reviews compared to negative and neutral sentiments. With the results of this study, business managers and sellers can leverage their customers’ reviews to better understand their needs by providing better services and to increase their sales by improving the product titles, descriptions and features by using keywords or search terms that are more relevant.
![image](https://user-images.githubusercontent.com/58675575/178278861-6fd88e76-05e3-4c89-94a6-cd5f7c23e9d4.png)

|Topic|Terms per Topic|
|Topic 1|	taste, roast, flavor, try, blend, smell, buy, coffee, pretty, think, expect, better, say, brand, even, bit, look, aroma, prefer, give|
|Topic 2	|cup, pod, work, product, order, box, machine, receive, package, time, come, get, ground, make, buy, money, gift, purchase, problem, packaging|
|Topic 3	|price, flavor, love, buy, find, variety, try, brand, order, purchase, pack, cup, store, get, decaf, capsule, time, far, coffee, enjoy|
|Topic 4	|star, love, taste, product, price, best, thank, delivery, buy, stuff, value, order, discount, time, service, coconut, always, arrive, fast, quality|
|Topic 5	|flavor, love, taste, drink, morning, creamer, add, recommend, make, caffeine, day, sugar, enjoy, milk, tea, need, vanilla, cup, highly, mix|
|Topic 6	|bean, make, brew, use, bag, get, try, product, drink, water, grind, buy, find, go, review, time, espresso, know, year, take|


