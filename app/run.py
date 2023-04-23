import json
import plotly
import pandas as pd
import sqlite3

from models.train_classifier import tokenize as tkn

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    # call the function from the train_classifier.py
    return tkn(text)


# load data
# connect to the database
conn = sqlite3.connect('../data/DisasterResponse.db')
# run a query
df = pd.read_sql('SELECT * FROM disaster_response_messages', conn)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    top_10_categories_counts = df.iloc[:, 4:].sum(
    ).sort_values(ascending=False).head(10)
    category_names = list(top_10_categories_counts.index)

    all_words = []
    for message in df['message']:
        all_words.extend(tokenize(message))
    top_10_words = pd.Series(all_words).value_counts(ascending=False).head(10)
    words = list(top_10_words.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=top_10_categories_counts
                )
            ],

            'layout': {
                'title': 'Top 10 disaster message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=top_10_words
                )
            ],

            'layout': {
                'title': 'Top 10 disaster message words used',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
