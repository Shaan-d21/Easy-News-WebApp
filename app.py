# Import necessary libraries
from flask import Flask, render_template, request
import newspaper
import pandas as pd
import feedparser
from flask_sqlalchemy import SQLAlchemy
import pickle
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Initialize Flask app
app = Flask(__name__)

scheduler = BackgroundScheduler(daemon=True)
scheduler.start()

# Global variables for news data
news_data_list = []

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///newsapp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class NewsApp(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    publish_date = db.Column(db.DateTime, nullable=True)
    content = db.Column(db.Text)
    url = db.Column(db.Text)
    imageurl = db.Column(db.Text)
    sentiments = db.Column(db.String(10), nullable=False)
    
    def __repr__(self) ->str:
        return f"{self.sno} - {self.title}"

   

# with app.app_context():
#     db.create_all()

# Function to scrape news from a feed URL
def scrape_news_from_feed(feed_url):
    articles = []
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        article = newspaper.Article(entry.link)
        article.download()
        article.parse()
        article_data = {
            'title': article.title,
            'author': article.authors,
            'publish_date': article.publish_date,
            'content': article.text,
            'url': article.url,
            'imageurl': article.top_image
        }
        articles.append(article_data)
    return articles

def clean_data(df):
    # Replace 'NaT' values in publish_date with default datetime
    df['publish_date'] = df['publish_date'].fillna(datetime.now())
    return df
# Function to fetch news and perform sentiment analysis
def fetch_and_analyze_news(feed_url):
    global news_data_list
    articles = scrape_news_from_feed(feed_url)
    for article in articles:
        news_data_list.append(article)

    # Load sentiment analysis model
    news_model = pickle.load(open('models/newsSentimentModel.pkl', 'rb'))
    vect_model = pickle.load(open('models/tfidfVectorizerModel.pkl', 'rb'))

    # Transform news titles and predict sentiments
    df = pd.DataFrame(news_data_list, columns=['title', 'author', 'publish_date', 'content', 'url', 'imageurl'])
    newstitle = vect_model.transform(df['title'])
    sentiments = news_model.predict(newstitle)
    df['sentiments'] = sentiments

    return df

def store_dataframe(df):
    # Iterate through DataFrame rows and add them to the database session
     with app.app_context():
        for index, record in df.iterrows():
            existing_news = NewsApp.query.filter_by(url=record['url']).first()
            if not existing_news:
                author_str = ', '.join(record['author']) if isinstance(record['author'], list) else record['author']
                publish_date_str = record['publish_date'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(record['publish_date'], pd.Timestamp) else record['publish_date']
                news_entry = NewsApp(
                    title=record['title'],
                    author=author_str,
                    content=record['content'],
                    publish_date=record['publish_date'],
                    url=record['url'],
                    imageurl=record['imageurl'],
                    sentiments=record['sentiments']
                )
                db.session.add(news_entry)
        db.session.commit()
    
    
def run_periodic_task():
    feed_url = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
    news_df = fetch_and_analyze_news(feed_url)
    df_cleaned = clean_data(news_df)
    
    print("data stored 1")
    with app.app_context():
        store_dataframe(df_cleaned)

    feed_url = 'https://cfo.economictimes.indiatimes.com/rss/economy'
    news_df = fetch_and_analyze_news(feed_url)
    df_cleaned = clean_data(news_df)
    
    print("data stored 2")
    with app.app_context():
        store_dataframe(df_cleaned)

    feed_url = 'https://www.moneycontrol.com/rss/marketreports.xml'
    news_df = fetch_and_analyze_news(feed_url)
    df_cleaned = clean_data(news_df)
    
    print("data stored 3")
    with app.app_context():
        store_dataframe(df_cleaned)

# Schedule the task to run every 30 minutes
scheduler.add_job(run_periodic_task, trigger=IntervalTrigger(minutes=10))


# Route for home page
@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    news = NewsApp.query.order_by(NewsApp.sno.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('index.html', news=news)

@app.route('/positive-news')
def positive_news():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    positive_news = NewsApp.query.filter_by(sentiments='positive').order_by(NewsApp.publish_date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('positive.html', news=positive_news)

@app.route('/neutral-news')
def neutral_news():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    neutral_news =NewsApp.query.filter_by(sentiments='neutral').order_by(NewsApp.publish_date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('neutral.html', news=neutral_news)

@app.route('/negative-news')
def negative_news():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    negative_news = NewsApp.query.filter_by(sentiments='negative').order_by(NewsApp.publish_date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('negative.html', news=negative_news)

@app.route('/latest_news')
def latest_news():
    latest_news_items = NewsApp.query.order_by(NewsApp.id.desc()).limit(3).all()
    return render_template('latest_news.html', news=latest_news_items)

if __name__ == '__main__':
    app.run(debug=True)
