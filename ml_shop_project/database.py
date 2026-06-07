import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

DB_NAME = "reviews_store.db"

def init_db():
    """Инициализация базы данных и импорт данных из реального CSV-файла с Kaggle"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            sentiment INTEGER NOT NULL,
            product TEXT NOT NULL,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL
        )
    ''')
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM reviews")
    if cursor.fetchone()[0] == 0:
        csv_path = "Womens Clothing E-Commerce Reviews.csv"
        
        
        if os.path.exists(csv_path):
            print(f"Найдена таблица {csv_path}. Импортируем данные в SQLite...")
            
           
            df_kaggle = pd.read_csv(csv_path)
            
           
            df_kaggle = df_kaggle.dropna(subset=['Review Text', 'Rating'])
            
            
            df_slice = df_kaggle.head(500)
            
            data_to_insert = []
            for idx, row in df_slice.iterrows():
                text = str(row['Review Text'])
                rating = int(row['Rating'])
                
               
                sentiment = 1 if rating >= 4 else 0
                
              
                product = str(row['Class Name']) if pd.notna(row['Class Name']) else "General Clothing"
               
                user_id = f"user_{np.random.randint(100, 135)}" 
                
               
                random_days_ago = np.random.randint(0, 30)
                date_str = (datetime.now() - timedelta(days=random_days_ago)).strftime("%Y-%m-%d")
                
                data_to_insert.append((text, rating, sentiment, product, user_id, date_str))
                
           
            cursor.executemany('''
                INSERT INTO reviews (text, rating, sentiment, product, user_id, date) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            conn.commit()
            print(f"Успешно импортировано {len(data_to_insert)} строк из Kaggle в SQLite базу данных!")
        else:
            print(f"ОШИБКА: Файл '{csv_path}' не найден в папке {os.getcwd()}!")
            print("Пожалуйста, скачайте датасет с Kaggle и переименуйте его в 'clothing_reviews.csv'.")
            
    conn.close()

def get_all_reviews():
    """Получить все отзывы из базы данных в виде pandas DataFrame"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    return df

def add_review(text, rating, sentiment, product, user_id):
    """Добавить новый отзыв, оставленный пользователем через веб-интерфейс"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''
        INSERT INTO reviews (text, rating, sentiment, product, user_id, date) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (text, rating, sentiment, product, user_id, date_str))
    conn.commit()
    conn.close()