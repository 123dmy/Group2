import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class SentimentAnalysis:
    def __init__(self):
        self.df = None

    def load_data(self, path):
        """Load the dataset from the provided path."""
        self.df = pd.read_csv(path)

    def get_text_columns(self):
        """Get the text columns, calculate average length and unique entries."""
        text_columns = self.df.select_dtypes(include=['object'])  # Select text columns
        column_info = []

        for col in text_columns.columns:
            avg_len = text_columns[col].apply(len).mean()
            unique_entries = text_columns[col].nunique()
            column_info.append([col, avg_len, unique_entries])

        return pd.DataFrame(column_info, columns=['Column Name', 'Average Entry Length', 'Unique Entries'])

    def vader_sentiment_analysis(self, data):
        """Perform sentiment analysis using VADER."""
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []

        for text in data:
            score = analyzer.polarity_scores(text)['compound']
            scores.append(score)
            if score >= 0.05:
                sentiments.append('positive')
            elif score <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')

        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        """Perform sentiment analysis using TextBlob."""
        scores = []
        sentiments = []
        subjectivity_scores = []

        for text in data:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            scores.append(polarity)
            subjectivity_scores.append(subjectivity)

            if polarity > 0:
                sentiments.append('positive')
            elif polarity == 0:
                sentiments.append('neutral')
            else:
                sentiments.append('negative')

        return scores, sentiments, subjectivity_scores

    def distilbert_sentiment_analysis(self, data):
        """Perform sentiment analysis using DistilBERT."""
        if pipeline is None:
            raise ImportError("Transformers module is not installed.")

        distilbert_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        scores = []
        sentiments = []

        for text in data:
            result = distilbert_pipeline(text)[0]
            score = result['score']
            label = result['label']
            scores.append(score)

            if label in ['4 stars', '5 stars']:
                sentiments.append('positive')
            elif label == '3 stars':
                sentiments.append('neutral')
            else:
                sentiments.append('negative')

        return scores, sentiments


def conduct_sentiment_analysis(dataset):
    """Main function to perform sentiment analysis on the dataset."""
    sa = SentimentAnalysis()

    # Assume dataset is already loaded into the class
    sa.df = dataset

    # Get text columns and display them to the user
    text_columns_df = sa.get_text_columns()
    print("Text columns information:\n", text_columns_df)

    # Ask the user which column to analyze
    column_to_analyze = input("Please enter the column name you would like to analyze: ")

    # Verify if the column exists in the DataFrame
    if column_to_analyze not in sa.df.columns:
        print("Invalid column name. Please enter a valid column name from the list above.")
        return

    # Ask the user which sentiment analysis method to use
    print("Choose the sentiment analysis method:\n1. VADER\n2. TextBlob\n3. DistilBERT")
    analysis_choice = input("Enter the number of the method (1, 2, or 3): ")

    if analysis_choice == '1':
        scores, sentiments = sa.vader_sentiment_analysis(sa.df[column_to_analyze])
        result_df = pd.DataFrame({'Text': sa.df[column_to_analyze], 'Score': scores, 'Sentiment': sentiments})
    elif analysis_choice == '2':
        scores, sentiments, subjectivity = sa.textblob_sentiment_analysis(sa.df[column_to_analyze])
        result_df = pd.DataFrame(
            {'Text': sa.df[column_to_analyze], 'Score': scores, 'Sentiment': sentiments, 'Subjectivity': subjectivity})
    elif analysis_choice == '3':
        scores, sentiments = sa.distilbert_sentiment_analysis(sa.df[column_to_analyze])
        result_df = pd.DataFrame({'Text': sa.df[column_to_analyze], 'Score': scores, 'Sentiment': sentiments})
    else:
        print("Invalid choice.")
        return

    # Display the results
    print("\nSentiment analysis results:\n", result_df)


if __name__ == '__main__':
    # The following line can be removed if the function is called from another script
    main()




