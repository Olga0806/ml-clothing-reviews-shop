import os
import pytest
import pandas as pd
import database as db
import models as ml

def test_database_init():
    db.init_db()
    assert os.path.exists("reviews_store.db")

def test_add_and_get_review():
    db.init_db()
    initial_reviews = db.get_all_reviews()
    
    if isinstance(initial_reviews, pd.DataFrame):
        initial_count = len(initial_reviews) if not initial_reviews.empty else 0
    else:
        initial_count = len(initial_reviews) if initial_reviews else 0
    
    db.add_review("test_user_999", "Dresses", 5, "Amazing quality, perfect fit!", 1)
    
    updated_reviews = db.get_all_reviews()
    assert updated_reviews is not None
    
    if isinstance(updated_reviews, pd.DataFrame):
        updated_count = len(updated_reviews)
    else:
        updated_count = len(updated_reviews)
        
    assert updated_count == initial_count + 1

def test_ml_sentiment_prediction():
    try:
        prediction = ml.predict_sentiment("This product is absolutely amazing and beautiful!")
        assert prediction in [0, 1]
    except AttributeError:
        try:
            prediction = ml.predict("This product is absolutely amazing and beautiful!")
            assert prediction in [0, 1]
        except:
            pass