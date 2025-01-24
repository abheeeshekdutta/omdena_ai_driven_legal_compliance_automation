from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
# from rich import print

def analyze_text(text):
    """
    Analyze the text and return the results
    """
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text,
                           language='en')
    return results

def anonymize_text(text):
    """
    Anonymize the text and return the anonymized text
    """
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text,
                           language='en')
    anonymizer = AnonymizerEngine()
    anonymized_text = anonymizer.anonymize(text=text,analyzer_results=results)
    return anonymized_text

if __name__ == "__main__":

    text="My name is Abhishek Dutta. I live in Dublin, Ireland. I work as a Data Scientist at Google \
    and my phone number is +353 87 234 5678."

    print(analyze_text(text))
    print(anonymize_text(text))