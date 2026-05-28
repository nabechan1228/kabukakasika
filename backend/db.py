"""
データベース接続設定とORMモデル定義の一元管理モジュール。

全てのデータベースアクセスはこのモジュール経由で行うこと。
"""
import os
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------
# データベース接続設定
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stock_data.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------
# ORMモデル定義
# ---------------------------------------------------------
class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    code = Column(String, index=True)
    date = Column(String, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------
# セッション管理 (FastAPI Depends用)
# ---------------------------------------------------------
def get_db():
    """FastAPIのDependsで使用するDBセッションジェネレータ。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
