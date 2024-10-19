import json
import time
from itertools import chain
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Text,
    Tuple,
    Type,
    Union,
)

import duckdb
import jinja2
from rich.text import Text as RichText

import languru.exceptions
from languru.config import console, logger
from languru.exceptions import NotFound, NotSupported
from languru.types.openai_page import OpenaiPage
from languru.utils.common import chunks
from languru.utils.openai_utils import embeddings_create_with_cache, ensure_vector
from languru.utils.sql import (
    CREATE_EMBEDDING_INDEX_LINE,
    DISPLAY_SQL_PARAMS,
    DISPLAY_SQL_QUERY,
    display_sql_parameters,
    openapi_to_create_table_sql,
)

if TYPE_CHECKING:
    import pandas as pd
    from openai import OpenAI

    from languru.documents.document import Document, Point, PointWithScore, SearchResult
    from languru.types.rerank import RerankingObject

sql_stmt_show_tables = "SHOW TABLES"
sql_stmt_drop_table = "DROP TABLE IF EXISTS {{ table_name }} CASCADE"
sql_stmt_vector_search = dedent(
    """
    WITH vector_search AS (
        SELECT {{ columns_expr }}
        FROM {{ table_name }}
        ORDER BY relevance_score DESC
        LIMIT {{ top_k }}
    )
    SELECT p.*
    FROM vector_search p
    """
).strip()
sql_stmt_vector_search_with_documents = dedent(
    """
    WITH vector_search AS (
        SELECT {{ columns_expr }}
        FROM {{ table_name }}
        ORDER BY relevance_score DESC
        LIMIT {{ top_k }}
    )
    SELECT p.*, d.*
    FROM vector_search p
    JOIN {{ document_table_name }} d ON p.document_id = d.document_id
    ORDER BY p.relevance_score DESC
    """
).strip()
sql_stmt_remove_outdated_points = dedent(
    """
    DELETE FROM {{ table_name }} WHERE document_id = ? AND content_md5 != ?
    """
).strip()
sql_stmt_get_by_doc_ids = dedent(
    """
    SELECT {{ columns_expr }} FROM {{ table_name }} WHERE document_id IN ( {{ placeholders }} )
    """  # noqa: E501
).strip()


def show_tables(conn: "duckdb.DuckDBPyConnection") -> Tuple[Text, ...]:
    res: List[Tuple[Text]] = conn.sql(sql_stmt_show_tables).fetchall()
    return tuple(r[0] for r in res)


def install_vss_extension(conn: "duckdb.DuckDBPyConnection") -> None:
    conn.sql("INSTALL vss;")
    conn.sql("LOAD vss;")


def install_json_extension(conn: "duckdb.DuckDBPyConnection") -> None:
    conn.sql("INSTALL json;")
    conn.sql("LOAD json;")


def install_all_extensions(conn: "duckdb.DuckDBPyConnection") -> None:
    install_vss_extension(conn=conn)
    install_json_extension(conn=conn)


class PointQuerySet:
    def __init__(
        self,
        model: Type["Point"],
        *args,
        **kwargs,
    ):
        self.model = model
        self.__args = args
        self.__kwargs = kwargs

    def touch(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        drop: bool = False,
        force: bool = False,
        debug: bool = False,
    ) -> bool:
        """
        Create or update the database table for storing Point objects.

        This method ensures that the necessary table structure exists in the database
        for storing Point objects. It can also drop and recreate the table if needed.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        drop : bool, optional
            If True, drop the existing table before creating a new one.
            Default is False.
        force : bool, optional
            If True, force the operation even if it might result in data loss.
            This is required when dropping tables. Default is False.
        debug : bool, optional
            If True, print debug information including SQL queries.
            Default is False.

        Returns
        -------
        bool
            True if the table was successfully created or already exists,
            False otherwise.

        Notes
        -----
        - This method installs necessary extensions (JSON and VSS) before
        creating the table.
        - The table structure is based on the Point model's JSON schema.
        - An HNSW index is created on the embedding column for efficient
        similarity search.

        See Also
        --------
        drop : Method to drop the table.
        model_json_schema : Method to get the JSON schema of the Point model.
        """

        time_start = time.perf_counter() if debug else None

        # Install JSON and VSS extensions
        install_all_extensions(conn=conn)

        # Check if table exists
        is_table_exists = self.model.TABLE_NAME in show_tables(conn=conn)
        if is_table_exists and not drop:
            return True
        elif is_table_exists and drop:
            self.drop(conn=conn, force=force, debug=debug)

        # Install JSON and VSS extensions
        install_all_extensions(conn=conn)

        # Create table
        create_table_sql = openapi_to_create_table_sql(
            self.model.model_json_schema(),
            table_name=self.model.TABLE_NAME,
            primary_key="point_id",
            indexes=["document_id", "content_md5"],
        ).strip()
        create_table_sql = (
            create_table_sql
            + "\nSET hnsw_enable_experimental_persistence = true;\n"  # Required for HNSW index  # noqa: E501
            + CREATE_EMBEDDING_INDEX_LINE.format(
                table_name=self.model.TABLE_NAME,
                column_name="embedding",
                metric="cosine",
            )
        ).strip()

        if debug:
            console.print(
                f"\nCreating table: '{self.model.TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )
            # CREATE TABLE points (
            #     point_id TEXT,
            #     document_id TEXT NOT NULL,
            #     content_md5 TEXT NOT NULL,
            #     embedding FLOAT[512],
            #     PRIMARY KEY (point_id)
            # );
            # CREATE INDEX idx_points_embedding ON points USING HNSW(embedding) WITH (metric = 'cosine');  # noqa: E501

        conn.sql(create_table_sql)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Created table: '{self.model.TABLE_NAME}' in {time_elapsed:.6f} ms"
            )
        return True

    def retrieve(
        self,
        point_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
        with_embedding: bool = False,
    ) -> "Point":
        """
        Retrieve a single Point object from the database by its ID.

        This method fetches a Point object from the database using its unique
        point_id. It can optionally include or exclude the embedding data.

        Parameters
        ----------
        point_id : Text
            The unique identifier of the Point to retrieve.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries.
            Default is False.
        with_embedding : bool, optional
            If True, include the embedding data in the retrieved Point.
            Default is False.

        Returns
        -------
        Point
            The retrieved Point object.

        Raises
        ------
        NotFound
            If no Point with the given point_id is found in the database.

        Notes
        -----
        - The method constructs a SQL query based on the Point model's schema.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        create : Method to create a new Point.
        update : Method to update an existing Point.
        """

        time_start = time.perf_counter() if debug else None

        # Get columns
        columns = list(self.model.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {self.model.TABLE_NAME} WHERE point_id = ?"
        parameters = [point_id]
        if debug:
            console.print(
                f"\nRetrieving point: '{point_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()

        if result is None:
            raise NotFound(f"Point with ID '{point_id}' not found.")

        data = dict(zip([c for c in columns], result))
        out = self.model.model_validate(data)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Retrieved point: '{point_id}' in {time_elapsed:.6f} ms")
        return out

    def create(
        self,
        point: Union["Point", Dict],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> "Point":
        """
        Create a single Point object in the database.

        This method creates a new Point object in the database. It can accept
        either a Point object or a dictionary representing a Point.

        Parameters
        ----------
        point : Union["Point", Dict]
            The Point object or dictionary representing the Point to be created.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries.
            Default is False.

        Returns
        -------
        Point
            The created Point object.

        Notes
        -----
        - This method internally calls the `bulk_create` method with a single point.
        - If a dictionary is provided, it will be validated against the Point model.

        See Also
        --------
        bulk_create : Method to create multiple Point objects at once.
        retrieve : Method to retrieve a Point object from the database.
        """

        points = self.bulk_create(points=[point], conn=conn, debug=debug)
        return points[0]

    def bulk_create(
        self,
        points: Union[
            Sequence["Point"], Sequence[Dict], Sequence[Union["Point", Dict]]
        ],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> List["Point"]:
        """
        Create multiple Point objects in the database.

        This method creates multiple Point objects in the database in a single operation.
        It can accept a sequence of Point objects, dictionaries, or a mix of both.

        Parameters
        ----------
        points : Union[Sequence["Point"], Sequence[Dict], Sequence[Union["Point", Dict]]]
            A sequence of Point objects or dictionaries representing Points to be created.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries.
            Default is False.

        Returns
        -------
        List["Point"]
            A list of created Point objects.

        Raises
        ------
        ValueError
            If any of the points are not embedded.

        Notes
        -----
        - This method validates all input points against the Point model.
        - It ensures that all points are embedded before insertion.
        - The method uses a single SQL INSERT statement for efficiency.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        create : Method to create a single Point object.
        touch : Method to ensure the Point table exists.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        if not points:
            return []
        points = [
            self.model.model_validate(p) if isinstance(p, Dict) else p for p in points
        ]
        for idx, pt in enumerate(points):
            if pt.is_embedded() is False:
                raise ValueError(
                    f"Points[{idx}] is not embedded, please embed it first."
                )

        # Get columns
        columns = list(points[0].model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: List[Tuple[Any, ...]] = []
        for pt in points:
            parameters.append(tuple([getattr(pt, c) for c in columns]))

        query = (
            f"INSERT INTO {self.model.TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )
        if debug:
            _display_params = display_sql_parameters(parameters)
            console.print(
                "\nCreating points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Create points
        conn.executemany(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Created {len(points)} points in {time_elapsed:.6f} ms")
        return points

    def update(self, *args, **kwargs):
        raise NotSupported("Updating points is not supported.")

    def remove(
        self,
        point_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """
        Remove a single Point object from the database by its ID.

        This method deletes a Point object from the database using its unique point_id.

        Parameters
        ----------
        point_id : Text
            The unique identifier of the Point to be removed.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries.
            Default is False.

        Returns
        -------
        None

        Notes
        -----
        - This method constructs and executes a DELETE SQL query.
        - Execution time is measured and printed if debug is True.
        - The method does not raise an exception if the point_id does not exist.

        See Also
        --------
        remove_many : Method to remove multiple Point objects at once.
        retrieve : Method to retrieve a Point object from the database.
        """

        time_start = time.perf_counter() if debug else None

        query = f"DELETE FROM {self.model.TABLE_NAME} WHERE point_id = ?"
        parameters = [point_id]
        if debug:
            console.print(
                f"\nDeleting point: '{point_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Deleted point: '{point_id}' in {time_elapsed:.6f} ms")
        return None

    def list(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        after: Optional[Text] = None,
        before: Optional[Text] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "asc",
        conn: "duckdb.DuckDBPyConnection",
        with_embedding: bool = False,
        debug: bool = False,
    ) -> OpenaiPage["Point"]:
        """
        List Point objects from the database with optional filtering and pagination.

        This method retrieves a list of Point objects from the database, with options
        for filtering by document_id or content_md5, pagination, and ordering.

        Parameters
        ----------
        document_id : Optional[Text], optional
            Filter points by document_id. Default is None.
        content_md5 : Optional[Text], optional
            Filter points by content_md5. Default is None.
        after : Optional[Text], optional
            Retrieve points after this point_id (for pagination). Default is None.
        before : Optional[Text], optional
            Retrieve points before this point_id (for pagination). Default is None.
        limit : int, optional
            Maximum number of points to retrieve. Default is 20.
        order : Literal["asc", "desc"], optional
            Order of retrieval ("asc" for ascending, "desc" for descending).
            Default is "asc".
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        with_embedding : bool, optional
            If True, include embedding data in the retrieved Points. Default is False.
        debug : bool, optional
            If True, print debug information including SQL queries. Default is False.

        Returns
        -------
        OpenaiPage["Point"]
            An OpenaiPage object containing the list of Point objects and pagination info.

        Notes
        -----
        - This method constructs and executes a SELECT SQL query with various conditions.
        - It supports pagination using the 'after' and 'before' parameters.
        - The method fetches one extra record to determine if there are more results.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        retrieve : Method to retrieve a single Point object.
        gen : Method to generate Point objects as an iterator.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        columns = list(self.model.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {self.model.TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if after is not None and order == "asc":
            where_clauses.append("point_id > ?")
            parameters.append(after)
        elif before is not None and order == "desc":
            where_clauses.append("point_id < ?")
            parameters.append(before)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        query += f"ORDER BY point_id {order.upper()}\n"

        fetch_limit = limit + 1
        query += f"LIMIT {fetch_limit}"

        if debug:
            console.print(
                "\nListing points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        results_df: "pd.DataFrame" = (
            conn.execute(query, parameters).fetch_arrow_table().to_pandas()
        )
        results: List[Dict] = results_df.to_dict(orient="records")

        points = [self.model.model_validate(row) for row in results[:limit]]

        out = OpenaiPage(
            data=points,
            object="list",
            first_id=points[0].point_id if points else None,
            last_id=points[-1].point_id if points else None,
            has_more=len(results) > limit,
        )

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Listed points in {time_elapsed:.6f} ms")
        return out

    def gen(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        after: Optional[Text] = None,
        before: Optional[Text] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "asc",
        conn: "duckdb.DuckDBPyConnection",
        with_embedding: bool = False,
        debug: bool = False,
    ) -> Generator["Point", None, None]:
        """
        Generate Point objects from the database with optional filtering and pagination.

        This method yields Point objects from the database, with options for filtering
        by document_id or content_md5, pagination, and ordering. It uses the `list` method
        internally to fetch points in batches.

        Parameters
        ----------
        document_id : Optional[Text], optional
            Filter points by document_id. Default is None.
        content_md5 : Optional[Text], optional
            Filter points by content_md5. Default is None.
        after : Optional[Text], optional
            Retrieve points after this point_id (for pagination). Default is None.
        before : Optional[Text], optional
            Retrieve points before this point_id (for pagination). Default is None.
        limit : int, optional
            Maximum number of points to retrieve per batch. Default is 20.
        order : Literal["asc", "desc"], optional
            Order of retrieval ("asc" for ascending, "desc" for descending).
            Default is "asc".
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        with_embedding : bool, optional
            If True, include embedding data in the retrieved Points. Default is False.
        debug : bool, optional
            If True, print debug information including SQL queries. Default is False.

        Yields
        ------
        Point
            Point objects retrieved from the database.

        Notes
        -----
        - This method uses the `list` method internally to fetch points in batches.
        - It continues to yield points until all matching points have been retrieved.
        - The method automatically handles pagination by updating the 'after' parameter.

        See Also
        --------
        list : Method to retrieve a list of Point objects.
        retrieve : Method to retrieve a single Point object.
        """  # noqa: E501

        has_more = True
        after = None
        while has_more:
            points = self.list(
                document_id=document_id,
                content_md5=content_md5,
                after=after,
                before=before,
                limit=limit,
                order=order,
                conn=conn,
                with_embedding=with_embedding,
                debug=debug,
            )
            has_more = points.has_more
            after = points.last_id
            for pt in points.data:
                yield pt
        return None

    def count(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> int:
        """
        Count the number of Point objects in the database, optionally filtered.

        This method counts the number of Point objects in the database, with optional
        filtering by document_id or content_md5.

        Parameters
        ----------
        document_id : Optional[Text], optional
            Filter points by document_id. Default is None.
        content_md5 : Optional[Text], optional
            Filter points by content_md5. Default is None.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries. Default is False.

        Returns
        -------
        int
            The number of Point objects matching the specified criteria.

        Notes
        -----
        - This method constructs and executes a COUNT SQL query.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        list : Method to retrieve a list of Point objects.
        retrieve : Method to retrieve a single Point object.
        """

        time_start = time.perf_counter() if debug else None

        query = f"SELECT COUNT(*) FROM {self.model.TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        if debug:
            console.print(
                "\nCounting points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()
        count = result[0] if result else 0

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Counted points in {time_elapsed:.6f} ms")
        return count

    def drop(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        force: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Drop the Point table from the database.

        This method drops the Point table from the database, removing all stored Point objects.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        force : bool, optional
            If True, force the drop operation. This is required to prevent accidental data loss.
            Default is False.
        debug : bool, optional
            If True, print debug information including SQL queries. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If force is False, to prevent accidental table drops.

        Notes
        -----
        - This method uses a SQL DROP TABLE statement to remove the table.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        touch : Method to create or update the Point table.
        """  # noqa: E501

        if not force:
            raise ValueError("Use force=True to drop table.")

        time_start = time.perf_counter() if debug else None

        query_template = jinja2.Template(sql_stmt_drop_table)
        query = query_template.render(table_name=self.model.TABLE_NAME)
        if debug:
            console.print(
                f"\nDropping table: '{self.model.TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        # Drop table
        conn.sql(query)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Dropped table: '{self.model.TABLE_NAME}' in {time_elapsed:.6f} ms"
            )
        return None

    def remove_outdated(
        self,
        *,
        document_id: Text,
        content_md5: Text,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """
        Remove outdated points (not matching content_md5) for a document.

        This method removes Point objects associated with a specific document_id
        that do not match the provided content_md5. This is useful for cleaning up
        outdated points when a document has been updated.

        Parameters
        ----------
        document_id : Text
            The ID of the document whose outdated points should be removed.
        content_md5 : Text
            The current content_md5 of the document. Points not matching this will be removed.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries. Default is False.

        Returns
        -------
        None

        Notes
        -----
        - This method uses a SQL DELETE statement to remove outdated points.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        remove : Method to remove a single Point object.
        remove_many : Method to remove multiple Point objects.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        query_template = jinja2.Template(sql_stmt_remove_outdated_points)
        query = query_template.render(table_name=self.model.TABLE_NAME)
        parameters = [document_id, content_md5]

        if debug:
            console.print(
                "\nRemoving outdated points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        # Remove outdated points
        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Deleted outdated points of document: '{document_id}' in "
                + f"{time_elapsed:.6f} ms"
            )
        return None

    def remove_many(
        self,
        point_ids: Optional[List[Text]] = None,
        *,
        document_ids: Optional[List[Text]] = None,
        content_md5s: Optional[List[Text]] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """
        Remove multiple Point objects from the database based on various criteria.

        This method removes Point objects from the database that match the specified
        point_ids, document_ids, or content_md5s.

        Parameters
        ----------
        point_ids : Optional[List[Text]], optional
            List of point IDs to remove. Default is None.
        document_ids : Optional[List[Text]], optional
            List of document IDs whose points should be removed. Default is None.
        content_md5s : Optional[List[Text]], optional
            List of content_md5 values whose points should be removed. Default is None.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, optional
            If True, print debug information including SQL queries. Default is False.

        Returns
        -------
        None

        Notes
        -----
        - This method constructs and executes a DELETE SQL query with multiple conditions.
        - At least one of point_ids, document_ids, or content_md5s must be provided.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        remove : Method to remove a single Point object.
        remove_outdated : Method to remove outdated points for a specific document.
        """  # noqa: E501

        if not any([point_ids, document_ids, content_md5s]):
            return None

        time_start = time.perf_counter() if debug else None

        query = f"DELETE FROM {self.model.TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if point_ids is not None:
            _placeholders = ", ".join(["?" for _ in point_ids])
            where_clauses.append(f"point_id IN ( {_placeholders} )")
            parameters.extend(point_ids)
        if document_ids is not None:
            _placeholders = ", ".join(["?" for _ in document_ids])
            where_clauses.append(f"document_id IN ( {_placeholders} )")
            parameters.extend(document_ids)
        if content_md5s is not None:
            _placeholders = ", ".join(["?" for _ in content_md5s])
            where_clauses.append(f"content_md5 IN ( {_placeholders} )")
            parameters.extend(content_md5s)

        if where_clauses:
            query += "WHERE " + " OR ".join(where_clauses) + "\n"

        if debug:
            console.print(
                "\nRemoving points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Deleted points in {time_elapsed:.6f} ms")
        return None


class DocumentQuerySet:
    def __init__(self, model: Type["Document"], *args, **kwargs):
        self.model = model
        self.__args = args
        self.__kwargs = kwargs

    def touch(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        touch_point: bool = True,
        drop: bool = False,
        force: bool = False,
        debug: bool = False,
    ) -> bool:
        """
        Create or update the Document table structure in the database.

        This method ensures that the Document table exists in the database with the
        correct schema. It can also optionally create or update the associated Point table.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        touch_point : bool, default True
            If True, also create or update the associated Point table.
        drop : bool, default False
            If True, drop existing tables before creating new ones.
        force : bool, default False
            If True, force the drop operation when `drop` is True.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        bool
            True if the table was created or updated, False if it already existed and
            was not modified.

        Notes
        -----
        - This method installs necessary DuckDB extensions.
        - It creates indexes on specified fields for better query performance.
        - If the table already exists and `drop` is False, the method returns without changes.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        drop : Method to drop the Document table.
        Point.objects.touch : Method to create or update the Point table.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        # Install JSON and VSS extensions
        install_all_extensions(conn=conn)

        # Check if table exists
        is_table_exists = self.model.TABLE_NAME in show_tables(conn=conn)
        if is_table_exists and drop:
            self.drop(conn=conn, force=force, drop_points=True, debug=debug)
        elif is_table_exists and not drop:
            return True

        # Install JSON and VSS extensions
        install_all_extensions(conn=conn)

        # Create table
        create_table_sql = openapi_to_create_table_sql(
            self.model.model_json_schema(),
            table_name=self.model.TABLE_NAME,
            primary_key="document_id",
            unique_fields=[],
            # unique_fields=["name"],  # Index limitations (https://duckdb.org/docs/sql/indexes)  # noqa: E501
            indexes=["content_md5"],
        )
        if debug:
            console.print(
                f"\nCreating table: '{self.model.TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )
            # CREATE TABLE documents (
            #     document_id TEXT,
            #     name VARCHAR(255) NOT NULL UNIQUE,
            #     content VARCHAR(5000) NOT NULL,
            #     content_md5 TEXT NOT NULL,
            #     metadata JSON,
            #     created_at INT,
            #     updated_at INT,
            #     PRIMARY KEY (document_id)
            # );
            # CREATE INDEX idx_documents_content_md5 ON documents (content_md5);

        conn.sql(create_table_sql)

        if touch_point:
            self.model.POINT_TYPE.objects.touch(conn=conn, debug=debug)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Created table: '{self.model.TABLE_NAME}' in {time_elapsed:.6f} ms"
            )
        return True

    def retrieve(
        self,
        document_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        with_embedding: bool = False,
        debug: bool = False,
    ) -> Optional["Document"]:
        """
        Retrieve a single Document object from the database by its document_id.

        This method fetches a Document from the database based on the provided document_id.

        Parameters
        ----------
        document_id : Text
            The unique identifier of the document to retrieve.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        with_embedding : bool, default False
            If True, include the embedding data in the retrieved Document.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        Optional["Document"]
            The retrieved Document object, or None if not found.

        Raises
        ------
        NotFound
            If the document with the specified document_id is not found in the database.

        Notes
        -----
        - This method constructs and executes a SELECT SQL query to fetch the document.
        - The metadata field is parsed from JSON to a Python dictionary.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        list : Method to retrieve multiple Document objects.
        create : Method to create a new Document object.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        columns = list(self.model.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = (
            f"SELECT {columns_expr} FROM {self.model.TABLE_NAME} WHERE document_id = ?"
        )
        parameters = [document_id]
        if debug:
            console.print(
                f"\nRetrieving document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()

        if result is None:
            raise NotFound(f"Document with ID '{document_id}' not found.")

        data = dict(zip(columns, result))
        data["metadata"] = json.loads(data["metadata"])
        out = self.model.model_validate(data)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Retrieved document: '{document_id}' in {time_elapsed:.6f} ms"
            )
        return out

    def create(
        self,
        document: Union["Document", Dict],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> "Document":
        """
        Create a new Document object in the database.

        This method inserts a new Document into the database. It can accept either
        a Document object or a dictionary representing the document.

        Parameters
        ----------
        document : Union["Document", Dict]
            The Document object or dictionary representing the document to be created.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        Document
            The created Document object.

        Notes
        -----
        - This method internally uses the `bulk_create` method to insert the document.
        - If a dictionary is provided, it is first validated and converted to a Document object.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        bulk_create : Method to create multiple Document objects at once.
        update : Method to update an existing Document object.
        """  # noqa: E501

        docs = self.bulk_create([document], conn=conn, debug=debug)
        return docs[0]

    def bulk_create(
        self,
        documents: Union[
            Sequence["Document"], Sequence[Dict], Sequence[Union["Document", Dict]]
        ],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> List["Document"]:
        """
        Create multiple Document objects in the database at once.

        This method inserts multiple Document objects into the database in a single operation.
        It can accept a sequence of Document objects, dictionaries, or a mix of both.

        Parameters
        ----------
        documents : Union[Sequence["Document"], Sequence[Dict], Sequence[Union["Document", Dict]]]
            A sequence of Document objects or dictionaries representing the documents to be created.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        List["Document"]
            A list of the created Document objects.

        Notes
        -----
        - This method validates and converts any dictionary inputs to Document objects.
        - It uses a single INSERT SQL statement to create all documents efficiently.
        - The method strips each document before insertion.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        create : Method to create a single Document object.
        update : Method to update an existing Document object.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        documents = [
            (
                self.model.model_validate(doc).strip()
                if isinstance(doc, Dict)
                else doc.strip()
            )
            for doc in documents
        ]

        columns = list(documents[0].model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: List[Tuple[Any, ...]] = [
            tuple(getattr(doc, c) for c in columns) for doc in documents
        ]

        query = (
            f"INSERT INTO {self.model.TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )
        if debug:
            _display_params = display_sql_parameters(parameters)
            console.print(
                "\nCreating documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Create documents
        conn.executemany(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Created {len(documents)} documents in {time_elapsed:.6f} ms"
            )
        return documents

    def update(
        self,
        document_id: Text,
        *,
        name: Optional[Text] = None,
        content: Optional[Text] = None,
        metadata: Optional[Dict] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> "Document":
        """
        Update an existing Document object in the database.

        This method updates the specified fields of a Document in the database.
        It can update the name, content, and/or metadata of the document.

        Parameters
        ----------
        document_id : Text
            The unique identifier of the document to update.
        name : Optional[Text], default None
            The new name for the document. If None, the name is not updated.
        content : Optional[Text], default None
            The new content for the document. If None, the content is not updated.
        metadata : Optional[Dict], default None
            A dictionary of metadata to update. If None, the metadata is not updated.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        Document
            The updated Document object.

        Raises
        ------
        ValueError
            If none of name, content, or metadata are provided for update.
            If the new name is already used by another document.
        NotFound
            If the document with the specified document_id is not found in the database.

        Notes
        -----
        - This method first checks if the document exists and if the new name (if provided) is unique.
        - It constructs and executes an UPDATE SQL query to modify the document.
        - The method updates the 'updated_at' timestamp automatically.
        - For metadata updates, it uses JSON merge patch to update only the specified fields.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        retrieve : Method to retrieve a Document object.
        create : Method to create a new Document object.
        """  # noqa: E501

        if not any([name, content, metadata]):
            raise ValueError("At least one of the parameters must be provided.")

        time_start = time.perf_counter() if debug else None

        # Check if the new name already exists
        if name is not None:
            existing_doc = conn.execute(
                "SELECT document_id FROM documents WHERE name = ? AND document_id != ?",
                [name, document_id],
            ).fetchone()
            if existing_doc:
                raise ValueError(
                    f"The name '{name}' is already used by another document."
                )

        document = self.retrieve(document_id, conn=conn)
        if document is None:
            raise languru.exceptions.NotFound(
                f"Document with ID '{document_id}' not found."
            )

        set_query: List[Text] = []
        parameters = []
        if name is not None:
            document.name = name
            set_query.append("name = ?")
            parameters.append(document.name)
        if content is not None:
            document.content = content
            document.strip()
            set_query.append("content = ?")
            parameters.append(document.content)
        if metadata is not None:
            document.metadata.update(metadata)
            set_query.append("metadata = json_merge_patch(metadata, ?::JSON)")
            parameters.append(json.dumps(metadata))
        document.updated_at = int(time.time())
        set_query.append("updated_at = ?")
        parameters.append(document.updated_at)

        set_query_expr = ",\n    ".join(set_query)
        parameters.append(document_id)
        query = f"UPDATE {self.model.TABLE_NAME}\n"
        query += f"SET {set_query_expr}\n"
        query += "WHERE document_id = ?"
        if debug:
            console.print(
                f"\nUpdating document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Updated document: '{document_id}' in {time_elapsed:.6f} ms")
        return document

    def remove(
        self,
        document_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        with_points: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Remove a Document object and optionally its associated Points from the database.

        This method deletes a Document from the database based on the provided document_id.
        It can also remove the associated Point objects if specified.

        Parameters
        ----------
        document_id : Text
            The unique identifier of the document to remove.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        with_points : bool, default True
            If True, also remove the associated Point objects for this document.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        None

        Notes
        -----
        - If `with_points` is True, it first removes all associated Point objects.
        - The method uses a DELETE SQL query to remove the document from the database.
        - If the document doesn't exist, the method completes without raising an error.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        Point.objects.remove_outdated : Method to remove associated Point objects.
        update : Method to update an existing Document object.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        # Remove points of the document
        if with_points:
            self.model.POINT_TYPE.objects.remove_outdated(
                document_id=document_id,
                content_md5="SHOULD_NOT_MATCHED",
                conn=conn,
                debug=debug,
            )

        # Prepare delete query
        query = f"DELETE FROM {self.model.TABLE_NAME} WHERE document_id = ?"
        parameters = [document_id]
        if debug:
            console.print(
                f"\nDeleting document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        # Delete document
        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Deleted document: '{document_id}' in {time_elapsed:.6f} ms")
        return None

    def list(
        self,
        *,
        after: Optional[Text] = None,
        before: Optional[Text] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "asc",
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> OpenaiPage["Document"]:
        """
        Retrieve a paginated list of documents from the database.

        This method fetches a list of documents, supporting pagination and ordering.

        Parameters
        ----------
        after : Optional[Text], default None
            The document_id to start fetching from (exclusive) when ordering ascending.
        before : Optional[Text], default None
            The document_id to start fetching from (exclusive) when ordering descending.
        limit : int, default 20
            The maximum number of documents to return.
        order : Literal["asc", "desc"], default "asc"
            The order in which to return the documents.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        OpenaiPage["Document"]
            A paginated object containing the list of documents and pagination metadata.

        Notes
        -----
        - The method uses SQL queries to fetch documents from the database.
        - It supports forward and backward pagination using 'after' and 'before' parameters.
        - The returned OpenaiPage object includes information about whether there are more results.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        OpenaiPage : The pagination object used to return results.
        count : Method to count the number of documents in the database.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        columns = list(self.model.model_json_schema()["properties"].keys())
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {self.model.TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if after is not None and order == "asc":
            where_clauses.append("document_id > ?")
            parameters.append(after)
        elif before is not None and order == "desc":
            where_clauses.append("document_id < ?")
            parameters.append(before)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        query += f"ORDER BY document_id {order.upper()}\n"

        # Fetch one more than the limit to determine if there are more results
        fetch_limit = limit + 1
        query += f"LIMIT {fetch_limit}"

        if debug:
            console.print(
                "\nListing documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        results_df: "pd.DataFrame" = (
            conn.execute(query, parameters).fetch_arrow_table().to_pandas()
        )
        results_df["metadata"] = results_df["metadata"].apply(json.loads)
        results: List[Dict] = results_df.to_dict(orient="records")

        documents = [self.model.model_validate(row) for row in results[:limit]]

        out = OpenaiPage(
            data=documents,
            object="list",
            first_id=documents[0].document_id if documents else None,
            last_id=documents[-1].document_id if documents else None,
            has_more=len(results) > limit,
        )

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Listed documents in {time_elapsed:.6f} ms")
        return out

    def count(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> int:
        """
        Count the number of documents in the database, optionally filtered by criteria.

        This method returns the count of documents, which can be filtered by document_id
        or content_md5.

        Parameters
        ----------
        document_id : Optional[Text], default None
            If provided, count only documents with this specific document_id.
        content_md5 : Optional[Text], default None
            If provided, count only documents with this specific content_md5.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        int
            The number of documents matching the specified criteria.

        Notes
        -----
        - The method uses a SQL COUNT query to determine the number of documents.
        - Filtering can be done using document_id, content_md5, or both.
        - If no filters are provided, it returns the total count of documents in the table.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        list : Method to retrieve a list of documents.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        query = f"SELECT COUNT(*) FROM {self.model.TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        if debug:
            console.print(
                "\nCounting documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()
        count = result[0] if result else 0

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Counted documents in {time_elapsed:.6f} ms")
        return count

    def drop(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        force: bool = False,
        drop_points: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Drop the document table and optionally the associated points table from the database.

        This method removes the document table from the database. It can also remove
        the associated points table if specified.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        force : bool, default False
            If True, forces the drop operation. This is required to prevent accidental
            data loss.
        drop_points : bool, default True
            If True, also drops the associated points table.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `force` is False, to prevent accidental table drops.

        Notes
        -----
        - This operation is irreversible and will result in data loss.
        - The method uses SQL templates to generate the DROP TABLE statements.
        - Execution time is measured and printed if debug is True.

        See Also
        --------
        touch : Method to create or update the table structure.
        """  # noqa: E501

        if not force:
            raise ValueError("Use force=True to drop table.")

        time_start = time.perf_counter() if debug else None

        query_template = jinja2.Template(sql_stmt_drop_table)
        query = query_template.render(table_name=self.model.TABLE_NAME)
        if debug:
            console.print(
                f"\nDropping table: '{self.model.TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        # Drop table
        conn.sql(query)

        # Drop points table
        if drop_points:
            self.model.POINT_TYPE.objects.drop(conn=conn, force=force, debug=debug)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Dropped table: '{self.model.TABLE_NAME}' in {time_elapsed:.6f} ms"
            )
        return None

    def documents_to_points(
        self,
        documents: Sequence["Document"],
        *,
        openai_client: Optional["OpenAI"] = None,
        batch_size: int = 500,
        debug: bool = False,
    ) -> List[Tuple["Point", ...]]:
        """
        Convert a sequence of Document objects to their corresponding Point objects.

        This method processes a sequence of Document objects and generates Point
        objects for each document, optionally creating embeddings for these points.

        Parameters
        ----------
        documents : Sequence["Document"]
            A sequence of Document objects to be converted to points.
        openai_client : Optional["OpenAI"], optional
            An OpenAI client instance for generating embeddings. If None, embeddings
            will not be generated.
        batch_size : int, default 500
            The number of points to process in each batch when creating embeddings.
        debug : bool, default False
            If True, enable debug output for additional information during processing.

        Returns
        -------
        List[Tuple["Point", ...]]
            A list where each element is a tuple of Point objects corresponding to
            a single Document.

        Notes
        -----
        - Each Document is converted to one or more Point objects.
        - If an OpenAI client is provided, embeddings are generated for the points
        in batches.
        - The method uses caching for efficient embedding creation.
        - This method does not persist the points to the database; it only creates
        the Point objects in memory.

        See Also
        --------
        Document.to_points : Method used to convert a single Document to Points.
        embeddings_create_with_cache : Utility function for creating embeddings with caching.
        """  # noqa: E501

        output_doc_points: List[Tuple["Point", ...]] = []
        points_with_doc_cards: List[Tuple["Point", Text]] = []

        for _doc in documents:
            _doc.strip()
            _pts_with_doc_cards = _doc.to_points(debug=debug, with_document_card=True)
            output_doc_points.append(tuple([_pt for _pt, _ in _pts_with_doc_cards]))
            points_with_doc_cards.extend(_pts_with_doc_cards)

        # Batch create embeddings for points
        if openai_client is not None:
            for batched_pts_with_doc_cards in chunks(points_with_doc_cards, batch_size):
                embeddings = embeddings_create_with_cache(
                    input=[doc_card for _, doc_card in batched_pts_with_doc_cards],
                    model=self.model.POINT_TYPE.EMBEDDING_MODEL,
                    dimensions=self.model.POINT_TYPE.EMBEDDING_DIMENSIONS,
                    openai_client=openai_client,
                    cache=self.model.POINT_TYPE.embedding_cache(
                        self.model.POINT_TYPE.EMBEDDING_MODEL
                    ),
                )
                if len(embeddings) != len(batched_pts_with_doc_cards):
                    raise ValueError(
                        "Number of embeddings does not match number of points."
                    )
                for pt_with_doc_card, embedding in zip(
                    batched_pts_with_doc_cards, embeddings
                ):
                    pt_with_doc_card[0].embedding = embedding

        return output_doc_points

    def documents_sync_points(
        self,
        documents: Sequence["Document"],
        *,
        conn: "duckdb.DuckDBPyConnection",
        openai_client: "OpenAI",
        with_embeddings: bool = False,
        force: bool = False,
        debug: bool = False,
    ) -> List[Tuple["Point", ...]]:
        """
        Synchronize points for a sequence of documents in the database.

        This method ensures that the points associated with the given documents
        are up-to-date in the database. It creates new points, updates existing ones,
        and removes outdated points as necessary.

        Parameters
        ----------
        documents : Sequence["Document"]
            A sequence of Document objects to synchronize points for.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        openai_client : OpenAI
            An OpenAI client instance for generating embeddings.
        with_embeddings : bool, default False
            If True, include embeddings in the synchronized points.
        force : bool, default False
            If True, force synchronization even if points are already up-to-date.
        debug : bool, default False
            If True, print debug information including SQL queries and execution times.

        Returns
        -------
        List[Tuple["Point", ...]]
            A list where each element is a tuple of Point objects corresponding to
            a single Document.

        Raises
        ------
        ValueError
            If the number of documents exceeds 500 or if no points are created for a document.

        Notes
        -----
        - The method efficiently handles batch processing of documents.
        - It checks for existing points, creates new ones, and removes outdated points.
        - Embeddings are generated using the provided OpenAI client if with_embeddings is True.
        - The method ensures that all points are consistent with their corresponding documents.

        See Also
        --------
        documents_to_points : Method used to convert Documents to Points.
        Point.objects.bulk_create : Method used for bulk creation of Points.
        Point.objects.remove_many : Method used for removing outdated Points.
        """  # noqa: E501

        time_start = time.perf_counter() if debug else None

        if not documents:
            return []
        if len(documents) > 500:
            raise ValueError("Batch sync documents is limited to 500 documents now.")

        out: List[Tuple["Point", ...]] = [()] * len(documents)

        docs_ids_to_idx_map: Dict[Text, int] = {}
        up_to_date_doc_ids: Set[Text] = set()
        existing_outdated_points: Dict[int, List["Point"]] = {}
        for idx, doc in enumerate(documents):
            doc.strip()
            docs_ids_to_idx_map[doc.document_id] = idx
            existing_outdated_points[idx] = []

        # Collect get point query
        point_columns = list(
            self.model.POINT_TYPE.model_json_schema()["properties"].keys()
        )
        if not with_embeddings:
            point_columns = [c for c in point_columns if c != "embedding"]
        point_columns_expr = ",".join(point_columns)
        query_by_doc_ids_template = jinja2.Template(sql_stmt_get_by_doc_ids)
        query_by_doc_ids = query_by_doc_ids_template.render(
            table_name=self.model.POINT_TYPE.TABLE_NAME,
            columns_expr=point_columns_expr,
            placeholders=", ".join(["?" for _ in documents]),
        )
        params_by_doc_ids = [doc.document_id for doc in documents]
        if debug:
            _display_params = display_sql_parameters(params_by_doc_ids)
            console.print(
                "\nGetting points by document IDs with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query_by_doc_ids)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Get points
        for item in (
            conn.execute(query_by_doc_ids, params_by_doc_ids)
            .fetch_arrow_table()
            .to_pandas()
            .to_dict(orient="records")
        ):
            _pt = self.model.POINT_TYPE.model_validate(item)
            _doc_idx = docs_ids_to_idx_map[_pt.document_id]
            if force is True:
                existing_outdated_points[_doc_idx].append(_pt)
            elif _pt.content_md5 != documents[_doc_idx].content_md5:
                existing_outdated_points[_doc_idx].append(_pt)
            else:
                up_to_date_doc_ids.add(_pt.document_id)
                out[_doc_idx] = tuple([_pt])

        # Create missing points
        required_created_points_docs = [
            doc for doc in documents if doc.document_id not in up_to_date_doc_ids
        ]
        if required_created_points_docs:
            created_points = self.documents_to_points(
                required_created_points_docs, openai_client=openai_client, debug=debug
            )
            self.model.POINT_TYPE.objects.bulk_create(
                list(chain.from_iterable(created_points)),
                conn=conn,
                debug=debug,
            )
            for _doc, _pts in zip(required_created_points_docs, created_points):
                if with_embeddings is False:
                    for _pt in _pts:
                        _pt.embedding = []
                out[docs_ids_to_idx_map[_doc.document_id]] = tuple(_pts)

        # Remove outdated points
        self.model.POINT_TYPE.objects.remove_many(
            point_ids=list(
                chain.from_iterable(
                    pt.point_id
                    for pts in existing_outdated_points.values()
                    for pt in pts
                )
            ),
            conn=conn,
            debug=debug,
        )

        # Validate output
        for idx, (_doc, _pts) in enumerate(zip(documents, out)):
            if len(_pts) == 0:
                raise ValueError(f"No points created for documents[{idx}]")
            for _pt in _pts:
                if (
                    _pt.document_id != _doc.document_id
                    or _pt.content_md5 != _doc.content_md5
                ):
                    raise ValueError(
                        "Document ID or Content MD5 does not match: "
                        + f"{_doc=}, {_pts=}"
                    )

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Synced points in {time_elapsed:.2f} ms")
        return out

    def search(
        self,
        query: Text | List[float],
        *,
        conn: "duckdb.DuckDBPyConnection",
        openai_client: Optional["OpenAI"] = None,
        rerank_client: Optional["OpenAI"] = None,
        top_k: int = 100,
        with_embedding: bool = False,
        with_documents: bool = False,
        debug: bool = False,
    ) -> "SearchResult":
        """
        Perform a semantic search on the documents using the given query.

        This method conducts a vector search on the document points and optionally
        reranks the results using a separate reranking model.

        Parameters
        ----------
        query : Text | List[float]
            The search query. Can be either a text string or a pre-computed
            embedding vector.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        openai_client : OpenAI, optional
            The OpenAI client to use for generating embeddings if the query
            is a text string.
        rerank_client : OpenAI, optional
            The OpenAI client to use for reranking results. If provided,
            results will be reranked using this client's rerank method.
        top_k : int, default 100
            The number of top results to return.
        with_embedding : bool, default False
            If True, include the embedding vectors in the returned results.
        with_documents : bool, default False
            If True, include the full document objects in the returned results.
        debug : bool, default False
            If True, print debug information including SQL queries and
            execution times.

        Returns
        -------
        SearchResult
            An object containing the search results, including matched points,
            optionally full documents, and metadata about the search operation.

        Notes
        -----
        - If the query is a text string, it will be converted to an embedding
        vector using the openai_client.
        - The method first performs a vector search and then optionally reranks
        the results if a rerank_client is provided.
        - Execution time is measured and included in the returned SearchResult.

        See Also
        --------
        search_vector : Method to perform vector search without text-to-embedding
                        conversion or reranking.
        """

        from languru.documents.document import SearchResult

        time_start = time.perf_counter() if debug else None

        # Collect query vector
        vector = ensure_vector(
            query,
            openai_client=openai_client,
            cache=self.model.POINT_TYPE.embedding_cache(
                self.model.POINT_TYPE.EMBEDDING_MODEL
            ),
            input_func=self.model.to_query_cards,
            embedding_model=self.model.POINT_TYPE.EMBEDDING_MODEL,
            embedding_dimensions=self.model.POINT_TYPE.EMBEDDING_DIMENSIONS,
        )

        # Run vector search
        _search_time_start = time.perf_counter()
        points_with_score, documents = self.search_vector(
            vector,
            conn=conn,
            top_k=top_k,
            with_embedding=with_embedding,
            with_documents=with_documents,
            debug=debug,
        )

        # Rerank results
        if rerank_client is not None:
            rerank_func = getattr(rerank_client.embeddings, "rerank", None)
            if rerank_func is None:
                logger.warning(
                    f"Rerank client {rerank_client} has no embeddings.rerank method."
                )
            elif isinstance(query, Text) is False:
                logger.warning("Query must be a string to rerank.")
            elif not documents:
                logger.warning("Documents must be provided to rerank.")
            else:
                logger.debug(f"Reranking {len(documents)} results with query: {query}")
                doc_id_scores: Dict[Text, float] = {}
                for pt in points_with_score:
                    if pt.document_id not in doc_id_scores:
                        doc_id_scores[pt.document_id] = pt.relevance_score
                    else:
                        doc_id_scores[pt.document_id] = max(
                            doc_id_scores[pt.document_id], pt.relevance_score
                        )

                documents_map = {
                    idx: doc.document_id for idx, doc in enumerate(documents)
                }
                rerank_obj: "RerankingObject" = rerank_func(
                    query=query,  # type: ignore
                    documents=[doc.content for doc in documents],
                )
                new_doc_id_scores: Dict[Text, float] = {
                    documents_map[result.index]: result.relevance_score
                    for result in rerank_obj.results
                }
                for point in points_with_score:
                    point.relevance_score = new_doc_id_scores[point.document_id]
                # Sort points_with_score by relevance_score in descending order
                points_with_score.sort(key=lambda x: x.relevance_score, reverse=True)
                # Sort documents by relevance_score in descending order
                documents.sort(
                    key=lambda x: new_doc_id_scores[x.document_id], reverse=True
                )
                if debug:
                    console.print("\nRerank results:")
                    console.print("=== Start of Rerank Results ===")
                    for doc in documents:
                        _content = RichText("[")
                        _content += RichText(doc.document_id, style="bright_cyan")
                        _content += RichText("]")
                        _content += RichText("(")
                        _content += RichText(
                            (
                                f"{doc.name[:29]}" + "..."
                                if len(doc.name) > 32
                                else f"{doc.name:<32}"
                            ),
                            style="bright_green",
                        )
                        _content += RichText("): ")
                        _content += RichText(
                            f"{doc_id_scores[doc.document_id]:.4f}", style="bright_blue"
                        )
                        _content += RichText(" --> ")
                        _content += RichText(
                            f"{new_doc_id_scores[doc.document_id]:.4f}",
                            style="bright_magenta",
                        )
                        console.print(_content)
                    console.print("==== End of Rerank Results ====\n")

        # Wrap results
        out = SearchResult.model_validate(
            {
                "query": query if isinstance(query, Text) else None,
                "matches": points_with_score,
                "documents": documents,
                "total_results": (
                    len(documents) if documents else len(points_with_score)
                ),
                "execution_time": (time.perf_counter() - _search_time_start),
                "relevance_score": (
                    points_with_score[0].relevance_score if points_with_score else 0.0
                ),
            }
        )

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Search execution time: {time_elapsed:.2f} ms")
        return out

    def search_vector(
        self,
        vector: List[float],
        *,
        conn: "duckdb.DuckDBPyConnection",
        top_k: int = 100,
        with_embedding: bool = False,
        with_documents: bool = False,
        debug: bool = False,
    ) -> Tuple[List["PointWithScore"], Optional[List["Document"]]]:
        """
        Perform a vector search on the document points using a given embedding vector.

        This method executes a similarity search in the vector space of document
        points, returning the most similar points and optionally the corresponding
        documents.

        Parameters
        ----------
        vector : List[float]
            The query vector to search against. Should have the same dimensionality
            as the stored point embeddings.
        conn : duckdb.DuckDBPyConnection
            The DuckDB connection object to use for database operations.
        top_k : int, default 100
            The number of top results to return.
        with_embedding : bool, default False
            If True, include the embedding vectors in the returned point results.
        with_documents : bool, default False
            If True, include the full document objects corresponding to the
            matched points.
        debug : bool, default False
            If True, print debug information including SQL queries and
            execution times.

        Returns
        -------
        Tuple[List[PointWithScore], Optional[List[Document]]]
            A tuple containing two elements:
            1. A list of PointWithScore objects representing the matched points
            and their similarity scores.
            2. If with_documents is True, a list of Document objects corresponding
            to the matched points; otherwise, None.

        Notes
        -----
        - The method uses cosine similarity for vector comparison.
        - Results are sorted by descending similarity score.
        - Execution time is measured if debug is True.

        See Also
        --------
        search : Higher-level method that handles text queries and reranking.
        """

        from languru.documents.document import PointWithScore

        time_start = time.perf_counter() if debug else None

        # Get point columns
        point_columns = list(
            self.model.POINT_TYPE.model_json_schema()["properties"].keys()
        )
        if not with_embedding:
            point_columns = [c for c in point_columns if c != "embedding"]
        point_columns += [
            "array_cosine_similarity("
            + f"embedding, ?::FLOAT[{self.model.POINT_TYPE.EMBEDDING_DIMENSIONS}]"
            + ") AS relevance_score"
        ]

        query_template = (
            jinja2.Template(sql_stmt_vector_search_with_documents)
            if with_documents
            else jinja2.Template(sql_stmt_vector_search)
        )
        query = query_template.render(
            table_name=self.model.POINT_TYPE.TABLE_NAME,
            document_table_name=self.model.TABLE_NAME,
            columns_expr=", ".join(point_columns),
            top_k=top_k,
        )
        parameters = [vector]

        if debug:
            _display_params = display_sql_parameters(parameters)
            console.print(
                "\nVector search with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query.strip())}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Execute query
        results_df: "pd.DataFrame" = (
            conn.execute(query, parameters).fetch_arrow_table().to_pandas()
        )
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]

        # Parse results
        points_with_score = []
        documents = []
        walked_docs = set()
        if with_documents:
            results_df["metadata"] = results_df["metadata"].apply(json.loads)
        rows: List[Dict] = results_df.to_dict(orient="records")
        for row in rows:
            points_with_score.append(PointWithScore.model_validate(row))
            if with_documents and row["document_id"] not in walked_docs:
                documents.append(self.model.model_validate(row))
                walked_docs.add(row["document_id"])

        # Log execution time
        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Vector search execution time: {time_elapsed:.2f} ms")

        return (points_with_score, documents)


class PointQuerySetDescriptor:
    def __get__(self, instance: None, owner: Type["Point"]) -> "PointQuerySet":
        if instance is not None:
            raise AttributeError(
                "PointQuerySetDescriptor cannot be accessed via an instance."
            )
        return PointQuerySet(owner)


class DocumentQuerySetDescriptor:
    def __get__(self, instance: None, owner: Type["Document"]) -> "DocumentQuerySet":
        if instance is not None:
            raise AttributeError(
                "DocumentQuerySetDescriptor cannot be accessed via an instance."
            )
        return DocumentQuerySet(owner)
