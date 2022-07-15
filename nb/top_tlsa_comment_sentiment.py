
import secrets_reddit
import random

from typing import Dict, List

from praw import Reddit
from praw.models.reddit.subreddit import Subreddit
from praw.models import MoreComments

from transformers import pipeline


def get_subreddit(display_name:str) -> Subreddit:
    """Get subreddit object from display name

    Args:
        display_name (str): Name of subreddit (community)

    Returns:
        Subreddit: Subreddit object
    """
    reddit = Reddit(
        client_id=secrets_reddit.REDDIT_API_CLIENT_ID,        
        client_secret=secrets_reddit.REDDIT_API_CLIENT_SECRET,
        user_agent=secrets_reddit.REDDIT_API_USER_AGENT
        )
    
    subreddit = reddit.subreddit(display_name) # YOUR CODE HERE
    return subreddit

def get_comments(subreddit:Subreddit, limit:int=3) -> List[str]:
    """ Get comments from subreddit

    Args:
        subreddit (Subreddit): Subreddit object
        limit (int, optional): Number of posts (submissions) to extract. Defaults to 3.

    Returns:
        List[str]: List of comments
    """
    top_comments = []
    for submission in subreddit.top(limit=limit):
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            top_comments.append(top_level_comment.body)
    return top_comments

def run_sentiment_analysis(comment:str) -> Dict:
    """Run sentiment analysis on comment using default distilbert model
    
    Args:
        comment (str): Comment
        
    Returns:
        str: Sentiment analysis result
    """
    sentiment_model = pipeline("sentiment-analysis")  # YOUR CODE HERE
    sentiment = sentiment_model(comment)
    return sentiment[0]


if __name__ == '__main__':
    subreddit = get_subreddit("TSLA")  # YOUR CODE HERE
    comments = get_comments(subreddit)
    comment = random.choice(comments)  # YOUR CODE HERE
    sentiment = run_sentiment_analysis(comment)
    
    print(f'The comment: {comment}')
    print(f'Predicted Label is {sentiment["label"]} and the score is {sentiment["score"]:.3f}')
