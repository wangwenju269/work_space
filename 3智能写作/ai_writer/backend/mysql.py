import sqlite3
import os 
from metagpt.const import DATA_PATH

class KnowledgeBaseManager:
    def __init__(self):
        self.database = os.path.join(DATA_PATH, "DB", "writer.db")
        self.create_tables_()

    def execute_query_(self, query, params, commit=False, fetch=False):
        conn = sqlite3.connect(self.database)
        # 启用外键约束
        conn.execute('PRAGMA foreign_keys = ON')
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            if commit:
                conn.commit()
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            conn.rollback()
            result = None
        finally:
            cursor.close()
            conn.close()

        return result

    def create_tables_(self):
        query = """
            CREATE TABLE IF NOT EXISTS User (
                user_id VARCHAR(255) PRIMARY KEY,
                user_name VARCHAR(255)
            );
        """
        self.execute_query_(query, (), commit=True)  
        
        query = """
            CREATE TABLE IF NOT EXISTS File (
                file_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                file_name VARCHAR(255),
                status VARCHAR(255),
                timestamp VARCHAR(255),
                file_path VARCHAR(512),
                persist_dir VARCHAR(512),
                deleted INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES User(user_id)
            );
        """
        self.execute_query_(query, (), commit=True)
        
        query = """
            CREATE TABLE IF NOT EXISTS OutLine (
                topic VARCHAR(255) PRIMARY KEY,
                requiment TEXT,
                outline TEXT
            );
        """
        self.execute_query_(query, (), commit=True)

        query = """
            CREATE TABLE IF NOT EXISTS Document (
                file_name VARCHAR(640),
                chapter_id  VARCHAR(64),
                instruction VARCHAR(64),
                name VARCHAR(64),
                content TEXT,
                PRIMARY KEY (file_name, chapter_id)
            );
        """
        self.execute_query_(query, (), commit=True)
    
    def add_user(self, user_id, user_name):
        query = "INSERT OR IGNORE INTO User (user_id, user_name) VALUES (?, ?)"
        self.execute_query_(query, (user_id, user_name), commit=True) 
        
    def add_file(self, user_id, file_id, file_name, status,  timestamp, file_path, persist_dir):
        self.add_user(user_id, "Default User")  # 你可以根据需要设置 user_name
        query = "INSERT INTO File (user_id, file_id, file_name, status, timestamp, file_path, persist_dir) VALUES (?, ?, ?, ?, ?, ?, ?)"
        self.execute_query_(query, (user_id, file_id, file_name, status, timestamp, file_path, persist_dir), commit=True)
    
    def add_outline(self, requiment, outline):
        query = "INSERT INTO OutLine (requiment, outline) VALUES (?, ?)"
        self.execute_query_(query, (requiment, outline), commit=True)
    
    def add_content(self, chapter_id, name, instruction, content):
        query = "INSERT INTO Document (chapter_id, name, instruction, content) VALUES (?, ?, ?, ?)"
        self.execute_query_(query, (chapter_id, name, instruction, content), commit=True)
    
    def get_persist_dir_by_file_id(self, file_id):
        query = "SELECT persist_dir FROM File WHERE file_id = ?"
        result = self.execute_query_(query, (file_id,), fetch=True)
        
        if result:
            return result[0][-1]  # 返回第一个结果的 persist_dir
        else:
            return None  # 如果没有找到对应的 file_id，返回 None 





























 

