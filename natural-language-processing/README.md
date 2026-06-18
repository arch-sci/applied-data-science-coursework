# NLP — Media Framing Analysis 

Project for the NLP course: analyzing how media outlets frame a real-world event differently depending on political orientation.

**Assignment:** — determine research question, scrape/collect dataset, NLP analysis (preprocessing, sentiment, topic modeling), 10-slide presentation, and report. Full details in `report.pdf`.

**Research question:** How do left-wing and right-wing English-language media differ in sentiment and topic framing when reporting on US actions against Nicolás Maduro?

**Approach:** Scraped 188 articles from 18 outlets (left/right classified via AllSides/Ad Fontes), then ran lexical bigram analysis, sentiment analysis (VADER baseline vs. RoBERTa transformer), NMF topic modeling, and spaCy-based Named Entity Recognition to compare framing across ideological lines.

## Stack

Python · pandas · scikit-learn (TF-IDF, NMF) · spaCy · NLTK (VADER) · HuggingFace Transformers (RoBERTa) · scipy (Mann-Whitney U) · matplotlib · seaborn · wordcloud
