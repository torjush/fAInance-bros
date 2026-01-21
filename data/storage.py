"""SQLite storage layer for the stock analyzer."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Storage:
    """SQLite storage for stock analysis data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Companies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    last_analyzed TIMESTAMP
                )
            """)

            # Price history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    UNIQUE(ticker, date)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_ticker_date
                ON price_history(ticker, date)
            """)

            # Insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    insight_type TEXT NOT NULL,
                    content TEXT,
                    summary TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_insights_ticker_time
                ON insights(ticker, timestamp)
            """)

            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    report_markdown TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_ticker_time
                ON reports(ticker, timestamp)
            """)

            # News cache table (to avoid re-fetching)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    url TEXT UNIQUE,
                    title TEXT,
                    published TIMESTAMP,
                    source TEXT,
                    extracted_data TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Filings cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS filings_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    url TEXT UNIQUE,
                    title TEXT,
                    published TIMESTAMP,
                    filing_type TEXT,
                    extracted_data TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info(f"Database initialized at {self.db_path}")

    # Company operations
    def upsert_company(self, ticker: str, name: str = None, sector: str = None):
        """Insert or update company info."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO companies (ticker, name, sector)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name = COALESCE(excluded.name, companies.name),
                    sector = COALESCE(excluded.sector, companies.sector)
            """, (ticker, name, sector))

    def get_company(self, ticker: str) -> dict | None:
        """Get company info."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM companies WHERE ticker = ?", (ticker,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_last_analyzed(self, ticker: str):
        """Update last_analyzed timestamp for a company."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE companies SET last_analyzed = ? WHERE ticker = ?
            """, (datetime.utcnow().isoformat(), ticker))

    def get_last_analyzed(self, ticker: str) -> datetime | None:
        """Get last analyzed timestamp for a company."""
        company = self.get_company(ticker)
        if company and company.get("last_analyzed"):
            return datetime.fromisoformat(company["last_analyzed"])
        return None

    # Price history operations
    def save_prices(self, ticker: str, prices: list[dict]):
        """Save price history data."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for price in prices:
                cursor.execute("""
                    INSERT INTO price_history (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, date) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume
                """, (
                    ticker,
                    price["date"],
                    price.get("open"),
                    price.get("high"),
                    price.get("low"),
                    price.get("close"),
                    price.get("volume"),
                ))
            logger.info(f"Saved {len(prices)} price records for {ticker}")

    def get_prices(
        self, ticker: str, start_date: str = None, end_date: str = None, limit: int = None
    ) -> list[dict]:
        """Get price history for a ticker."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM price_history WHERE ticker = ?"
            params = [ticker]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_price_date(self, ticker: str) -> str | None:
        """Get the most recent price date for a ticker."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) as max_date FROM price_history WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            return row["max_date"] if row else None

    # Insights operations
    def save_insight(
        self, ticker: str, insight_type: str, content: dict, summary: str
    ):
        """Save an insight."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO insights (ticker, insight_type, content, summary)
                VALUES (?, ?, ?, ?)
            """, (ticker, insight_type, json.dumps(content), summary))
            logger.info(f"Saved {insight_type} insight for {ticker}")

    def get_insights(
        self, ticker: str, insight_type: str = None, limit: int = 10
    ) -> list[dict]:
        """Get recent insights for a ticker."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM insights WHERE ticker = ?"
            params = [ticker]

            if insight_type:
                query += " AND insight_type = ?"
                params.append(insight_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]

            # Parse JSON content
            for row in rows:
                if row.get("content"):
                    try:
                        row["content"] = json.loads(row["content"])
                    except json.JSONDecodeError:
                        pass
            return rows

    # Reports operations
    def save_report(self, ticker: str, report_markdown: str):
        """Save a generated report."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (ticker, report_markdown)
                VALUES (?, ?)
            """, (ticker, report_markdown))
            logger.info(f"Saved report for {ticker}")

    def get_latest_report(self, ticker: str) -> dict | None:
        """Get the most recent report for a ticker."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM reports WHERE ticker = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (ticker,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # News cache operations
    def save_news_item(
        self,
        ticker: str,
        url: str,
        title: str,
        published: str,
        source: str,
        extracted_data: dict = None,
    ):
        """Save a news item to cache."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO news_cache (ticker, url, title, published, source, extracted_data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    extracted_data = COALESCE(excluded.extracted_data, news_cache.extracted_data)
            """, (
                ticker,
                url,
                title,
                published,
                source,
                json.dumps(extracted_data) if extracted_data else None,
            ))

    def get_cached_news(self, ticker: str, since: str = None) -> list[dict]:
        """Get cached news for a ticker."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM news_cache WHERE ticker = ?"
            params = [ticker]

            if since:
                query += " AND published >= ?"
                params.append(since)

            query += " ORDER BY published DESC"
            cursor.execute(query, params)

            rows = [dict(row) for row in cursor.fetchall()]
            for row in rows:
                if row.get("extracted_data"):
                    try:
                        row["extracted_data"] = json.loads(row["extracted_data"])
                    except json.JSONDecodeError:
                        pass
            return rows

    def news_exists(self, url: str) -> bool:
        """Check if a news item is already cached."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM news_cache WHERE url = ?", (url,))
            return cursor.fetchone() is not None

    # Filings cache operations
    def save_filing(
        self,
        ticker: str,
        url: str,
        title: str,
        published: str,
        filing_type: str = None,
        extracted_data: dict = None,
    ):
        """Save a filing to cache."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO filings_cache (ticker, url, title, published, filing_type, extracted_data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    extracted_data = COALESCE(excluded.extracted_data, filings_cache.extracted_data)
            """, (
                ticker,
                url,
                title,
                published,
                filing_type,
                json.dumps(extracted_data) if extracted_data else None,
            ))

    def get_cached_filings(self, ticker: str, since: str = None) -> list[dict]:
        """Get cached filings for a ticker."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM filings_cache WHERE ticker = ?"
            params = [ticker]

            if since:
                query += " AND published >= ?"
                params.append(since)

            query += " ORDER BY published DESC"
            cursor.execute(query, params)

            rows = [dict(row) for row in cursor.fetchall()]
            for row in rows:
                if row.get("extracted_data"):
                    try:
                        row["extracted_data"] = json.loads(row["extracted_data"])
                    except json.JSONDecodeError:
                        pass
            return rows

    def filing_exists(self, url: str) -> bool:
        """Check if a filing is already cached."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM filings_cache WHERE url = ?", (url,))
            return cursor.fetchone() is not None
