import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

def train_and_compare_models(df):
    """Обучает 3 модели на текстах отзывов и возвращает их метрики"""
  
    df_clean = df[df['sentiment'].notna()].copy()
    X = df_clean['text']
    y = df_clean['sentiment'].astype(int)
    
    
    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
   
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    metrics_results = []
    trained_models = {}
    
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_results.append({
            "Модель": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1-Score": round(f1_score(y_test, y_pred), 3)
        })
        trained_models[name] = model
        
    return pd.DataFrame(metrics_results), trained_models, vectorizer


def get_user_recommendations(df, target_user_id, top_n=3):
    """Рекомендательная система (User-Based Collaborative Filtering)"""
    
    user_item_matrix = df.pivot_table(index='user_id', columns='product', values='rating', aggfunc='mean')
    
    
    if target_user_id not in user_item_matrix.index:
       
        top_popular = df.groupby('product')['rating'].mean().sort_values(ascending=False).index.tolist()
        return top_popular[:top_n]
    
    
    matrix_filled = user_item_matrix.fillna(0)
    user_sim = cosine_similarity(matrix_filled)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
   
    similar_users = user_sim_df[target_user_id].sort_values(ascending=False).drop(target_user_id)
    
    
    target_user_ratings = user_item_matrix.loc[target_user_id]
    unrated_products = target_user_ratings[target_user_ratings.isna()].index
    
    predicted_ratings = {}
    for prod in unrated_products:
        other_ratings = user_item_matrix[prod].dropna()
        
        common_users = other_ratings.index.intersection(similar_users.index)
        
        if len(common_users) == 0:
            continue
            
        
        sim_weights = similar_users.loc[common_users]
        user_ratings = other_ratings.loc[common_users]
        
        if sim_weights.sum() > 0:
            predicted_ratings[prod] = np.dot(user_ratings, sim_weights) / sim_weights.sum()
            

    sorted_recs = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    recommended_products = [prod for prod, score in sorted_recs]
    
    
    if len(recommended_products) < top_n:
        global_top = df.groupby('product')['rating'].mean().sort_values(ascending=False).index.tolist()
        for p in global_top:
            if p not in recommended_products:
                recommended_products.append(p)
            if len(recommended_products) == top_n:
                break
                
    return recommended_products[:top_n]