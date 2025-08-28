from google import genai
from dotenv import load_dotenv
import os
import sqlite3
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Data class to hold query results and metadata."""
    sql: str
    results: List[Tuple]
    columns: List[str]
    explanation: str

    def __str__(self) -> str:
        # Format results nicely
        table = ""
        if self.results:
            header = " | ".join(self.columns)
            rows = "\n".join(" | ".join(str(v) for v in row) for row in self.results)
            table = f"\n{header}\n{rows}\n"
        else:
            table = "\nNo rows returned.\n"

        return (
            f"SQL used:\n{self.sql}\n\n"
            f"Query Results:{table}\n"
            f"Analysis:\n{self.explanation}\n"
        )


class HDBDataAnalyst:
    """
    A class for analyzing Singapore HDB BTO data using natural language queries.
    
    This class provides functionality to:
    - Convert natural language queries to SQL
    - Execute queries against HDB database
    - Generate analytical explanations of results
    """
    
    # SQL generation prompt template
    SQL_PROMPT_TEMPLATE = """
    You are a housing market intelligence assistant designed to support analysis of BTO (Build-To-Order) pricing in Singapore, leveraging trends from HDB resale transactions. Provide explanations and predictions for BTO prices based on past resale data and contextual factors.
    1. **Pricing Framework**
    - BTO flat prices are informed by nearby resale flat transactions: HDB benchmarks BTO “market value” by comparing transacted resale flat prices in the vicinity

    2. **Trends in BTO–Resale Price Gaps**
    - In **mature estates**, resale prices have risen faster than BTO prices
    - In **non-mature estates**, BTO price growth sometimes outpaces resale growth

    3. **Other Market Influences**
    - BTO supply levels significantly affect resale demand and prices. Over-supply of BTO units historically led to resale price stagnation, while undersupply—or demand exceeding supply—increases pressure on resale prices
   - Micro-level price drivers include: location/town attributes, estate maturity, flat type and size, remaining lease, floor level, and time of year. Empirical analyses confirm that higher floors,central/mature locations, larger flat sizes, newer builds (longer leases) command higher resale prices

    **Task Instructions**
    You are a SQL-only assistant for the HDB data mart (Singapore context). Given the above background information, your ONLY task is to understand the intent of the user query and emit **exactly one valid SQLite statement** (or a single CTE) that answers the user's question by reading from the tables below.

    Schema
    ------
    bto_prices(
        _id INTEGER PRIMARY KEY AUTOINCREMENT,
        financial_year TEXT,
        room_type TEXT,
        town TEXT,
        min_selling_price REAL,
        max_selling_price REAL,
        min_selling_price_less_ahg_shg REAL,
        max_selling_price_less_ahg_shg REAL
    )

    resale_prices (
        _id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT,
        town TEXT,
        flat_type TEXT,
        flat_model TEXT,
        block TEXT,
        street_name TEXT,
        storey_range TEXT,
        floor_area_sqm REAL,
        lease_commence_date TEXT,
        resale_price REAL
    )

    Sample data
    -----------
    {sample_data}
    

    Note that the data in the database is stored in lowercase

    Rules
    1. Return ONLY the SQL statement—no explanations, no markdown fences.  
    2. Use standard SQLite syntax (CTEs allowed).  
    3. Parameterise with $town, $room_type, etc. when possible.  
    4. Aggregate or filter as needed to answer the question; do NOT predict future prices.  
    5. If the question is ambiguous, choose the most reasonable interpretation and proceed.  
    6. The SQLite query must always be executable.

    Examples
    --------
    Q: List all towns with BTO launches in 2018.
    A:
    SELECT DISTINCT town
    FROM bto_prices
    WHERE financial_year = '2018';

    Q: {user_query}
    """

    # Analysis prompt template
    ANALYST_PROMPT_TEMPLATE = """
    You are a housing market intelligence assistant designed to support analysis of BTO (Build-To-Order) pricing in Singapore, leveraging trends from HDB resale transactions. Provide explanations and predictions for BTO prices based on past resale data and contextual factors.
    1. **Pricing Framework**
    - BTO flat prices are informed by nearby resale flat transactions: HDB benchmarks BTO “market value” by comparing transacted resale flat prices in the vicinity

    2. **Trends in BTO–Resale Price Gaps**
    - In **mature estates**, resale prices have risen faster than BTO prices
    - In **non-mature estates**, BTO price growth sometimes outpaces resale growth

    3. **Other Market Influences**
    - BTO supply levels significantly affect resale demand and prices. Over-supply of BTO units historically led to resale price stagnation, while undersupply—or demand exceeding supply—increases pressure on resale prices
    - Micro-level price drivers include: location/town attributes, estate maturity, flat type and size, remaining lease, floor level, and time of year. Empirical analyses confirm that higher floors,central/mature locations, larger flat sizes, newer builds (longer leases) command higher resale prices


    You will be given the following data and context:  
    1. A user query  
    2. An SQL query that extracts relevant information from the database  
    3. The corresponding SQL output  

    Your task is to response to the user’s query using the information provided. Understand the query, and provide analysis in natural language explanations and justifications.
    Ensure you always aim to respond to the user's query, but only use information provided.

    Schema
    ------
    bto_prices(
        _id INTEGER PRIMARY KEY AUTOINCREMENT,
        financial_year TEXT,
        room_type TEXT,
        town TEXT,
        min_selling_price REAL,
        max_selling_price REAL,
        min_selling_price_less_ahg_shg REAL,
        max_selling_price_less_ahg_shg REAL
    )

    resale_prices (
        _id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT,
        town TEXT,
        flat_type TEXT,
        flat_model TEXT,
        block TEXT,
        street_name TEXT,
        storey_range TEXT,
        floor_area_sqm REAL,
        lease_commence_date TEXT,
        resale_price REAL
    )

    Q: {user_query}
    SQL: {sql_query}
    Output: {output}
    """

    def __init__(self, db_path: str, model: str = "gemini-2.5-flash"):
        """
        Initialize the HDB Data Analyst.
        
        Args:
            db_path: Path to the SQLite database file
            model: Gemini model to use for query generation and analysis
        """
        load_dotenv()
        
        self.db_path = db_path
        self.model = model
        self._sample_rows_cache = self._sample_rows(["bto_prices", "resale_prices"], rows=2)

        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)

    def _sample_rows(self, tables: List[str], rows: int = 3) -> str:
        snippets = []
        with sqlite3.connect(self.db_path) as conn:
            for tbl in tables:
                try:
                    df = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT {rows};", conn)
                    if df.empty:
                        snippets.append(f"Table: {tbl}\n<empty>")
                    else:
                        snippets.append(f"Table: {tbl}\n{df.to_string(index=False)}")
                except Exception as e:
                    snippets.append(f"Table: {tbl} (error: {e})")
        return "\n\n".join(snippets)
    
    def _is_valid_sql(self, sql: str) -> bool:
        """
        Check if SQL query is valid by using EXPLAIN.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            True if SQL is valid, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"EXPLAIN {sql}")
            return True
        except sqlite3.Error as e:
            print(f"SQL validation error: {e}")
            return False
    
    def _generate_sql_query(self, user_query: str) -> str:
        """
        Generate SQL query from natural language query.
        
        Args:
            user_query: Natural language query
            
        Returns:
            Generated SQL query string
        """
        prompt = self.SQL_PROMPT_TEMPLATE.format(
                    user_query=user_query,
                    sample_data=self._sample_rows_cache
                )        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        
        return response.text.strip()
    
    def _generate_valid_sql(self, user_query: str, max_attempts: int = 3) -> str:
        """
        Generate a valid SQL query with retry logic.
        
        Args:
            user_query: Natural language query
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Valid SQL query string
            
        Raises:
            RuntimeError: If unable to generate valid SQL after max attempts
        """
        for attempt in range(max_attempts):
            sql = self._generate_sql_query(user_query)
            print(f"Generated SQL (attempt {attempt + 1}): {sql}")
            
            if self._is_valid_sql(sql):
                return sql
            
            print(f"SQL validation failed, retrying...")
        
        raise RuntimeError(f"Failed to generate valid SQL after {max_attempts} attempts")
    
    def _execute_sql(self, sql: str) -> Tuple[List[Tuple], List[str]]:
        """
        Execute SQL query and return results with column names.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Tuple of (results, column_names)
            
        Raises:
            sqlite3.Error: If query execution fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                results = cursor.fetchall()
                return results, columns
        except sqlite3.Error as e:
            print(f"Query execution error: {e}")
            raise
    
    def _generate_explanation(self, user_query: str, sql: str, results: List[Tuple]) -> str:
        """
        Generate analytical explanation of query results.
        
        Args:
            user_query: Original natural language query
            sql: SQL query that was executed
            results: Query results
            
        Returns:
            Analytical explanation string
        """
        result_str = str(results) if results else "No rows returned."
        
        prompt = self.ANALYST_PROMPT_TEMPLATE.format(
            user_query=user_query,
            sql_query=sql.strip(),
            output=result_str
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text.strip()
    
    def display_results(self, results: List[Tuple], columns: List[str]) -> None:
        """
        Display query results in a formatted table.
        
        Args:
            results: Query results
            columns: Column names
        """
        if not results:
            print("No results found.")
            return
        
        # Print header
        header = " | ".join(f"{col:<15}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in results:
            print(" | ".join(f"{str(val):<15}" for val in row))
    
    def query(self, user_query: str, display: bool = True) -> QueryResult:
        """
        Execute a natural language query and return comprehensive results.
        
        Args:
            user_query: Natural language query about HDB data
            display: Whether to print results to console
            
        Returns:
            QueryResult object containing SQL, results, columns, and explanation
        """
        # Generate and validate SQL
        sql = self._generate_valid_sql(user_query)
        
        # Execute query
        results, columns = self._execute_sql(sql)
        
        # Generate explanation
        explanation = self._generate_explanation(user_query, sql, results)
        
        return QueryResult(
            sql=sql,
            results=results,
            columns=columns,
            explanation=explanation
        )


def main():
    """Example usage of the HDBDataAnalyst class."""
    analyst = HDBDataAnalyst("data/hdb_prices.db")
    
    # Example query
    result = analyst.query("Please recommend housing estates that have had limited Build-To-Order (BTO) launches in the past 10 years")
    
    # Access individual components if needed
    print(result)


if __name__ == "__main__":
    main()