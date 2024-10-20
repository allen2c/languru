"""
This script compares different methods of executing vector similarity searches using DuckDB, focusing on performance and concurrency.

Key observations:
1. Single queries are generally fast, with new connections performing slightly better than existing ones.
2. Threading and thread pools show good performance, with new connections often outperforming existing ones.
3. Process pools are significantly slower due to connection creation overhead.
4. Async to thread performs well, with new connections showing a slight edge.

DuckDB insights:
- Handles concurrent queries efficiently, especially with new connections.
- New connections often perform better than existing ones in concurrent scenarios.

Conclusion:
For optimal performance in this scenario, using new connections with threading or thread pools appears to be the best approach. However, the ideal method may vary depending on specific use cases and system resources. Single queries and async to thread also show promising results.

[SingleQueryWithExistingConn] Time taken: 33.08 ms
[ThreadingQueryWithExistingConn-0] Time taken: 41.26 ms
[ThreadingQueryWithExistingConn-1] Time taken: 73.28 ms
[ThreadingQueryWithExistingConn-2] Time taken: 106.63 ms
[ThreadingPoolQueryWithExistingConn] Time taken: 37.33 ms
[ThreadingPoolQueryWithExistingConn] Time taken: 67.82 ms
[ThreadingPoolQueryWithExistingConn] Time taken: 98.42 ms
[AsyncToThreadQueryWithExistingConn-1] Time taken: 52.76 ms
[AsyncToThreadQueryWithExistingConn-0] Time taken: 86.87 ms
[AsyncToThreadQueryWithExistingConn-2] Time taken: 101.46 ms
[SingleQueryWithNewConn] Time taken: 37.12 ms
[ThreadingQueryWithNewConn-0] Time taken: 54.84 ms
[ThreadingQueryWithNewConn-1] Time taken: 57.25 ms
[ThreadingQueryWithNewConn-2] Time taken: 59.66 ms
[ThreadingPoolQueryWithNewConn] Time taken: 45.02 ms
[ThreadingPoolQueryWithNewConn] Time taken: 46.97 ms
[ThreadingPoolQueryWithNewConn] Time taken: 49.08 ms
[ProcessPoolQueryWithNewConn] Time taken: 440.76 ms
[ProcessPoolQueryWithNewConn] Time taken: 441.38 ms
[ProcessPoolQueryWithNewConn] Time taken: 441.89 ms
[AsyncToThreadQueryWithNewConn-0] Time taken: 60.85 ms
[AsyncToThreadQueryWithNewConn-2] Time taken: 63.56 ms
[AsyncToThreadQueryWithNewConn-1] Time taken: 65.63 ms
"""  # noqa: E501

import asyncio
import concurrent.futures
import threading
import time
from functools import partial
from textwrap import dedent
from typing import List, Text

import duckdb
import pyarrow as pa
from jinja2 import Template
from openai import OpenAI

from languru.documents.document import Document, Point
from languru.utils.openai_utils import embeddings_create_with_cache

DB_FILEPATH = "data/bbc-fulltext.db"
TOP_K = 1000

openai_client = OpenAI()

vss_template = Template(
    dedent(
        """
        WITH vector_search AS (
            SELECT point_id, document_id, content_md5, embedding, array_cosine_similarity(embedding, ?::FLOAT[{{ embedding_dimensions }}]) AS relevance_score
            FROM {{ points_table_name }}
            ORDER BY relevance_score DESC
            LIMIT {{ top_k }}
        )
        SELECT p.*, d.*
        FROM vector_search p
        JOIN {{ documents_table_name }} d ON p.document_id = d.document_id
        ORDER BY p.relevance_score DESC
        """  # noqa: E501
    ).strip()
)


def get_conn(
    db_filepath: Text, *, read_only: bool = False
) -> "duckdb.DuckDBPyConnection":
    return duckdb.connect(db_filepath, read_only=read_only)


def get_vectors(queries: List[Text]) -> List[List[float]]:
    return embeddings_create_with_cache(
        input=[q.strip() for q in queries],
        model=Point.EMBEDDING_MODEL,
        dimensions=Point.EMBEDDING_DIMENSIONS,
        openai_client=openai_client,
        cache=Point.embedding_cache(Point.EMBEDDING_MODEL),
    )


def search_conn_exec(
    *, conn: "duckdb.DuckDBPyConnection", vector: List[float], top_k: int
):
    vss_sql = vss_template.render(
        points_table_name=Point.TABLE_NAME,
        documents_table_name=Document.TABLE_NAME,
        embedding_dimensions=Point.EMBEDDING_DIMENSIONS,
        top_k=top_k,
    )
    results = conn.execute(vss_sql, [vector])
    arrow_table: "pa.Table" = results.fetch_arrow_table()
    return arrow_table


def single_query_with_new_conn(
    vector: List[float],
    *,
    silent: bool = False,
    timer_name: Text = "SingleQueryWithNewConn",
    read_only: bool = False,
):
    conn = get_conn(DB_FILEPATH, read_only=read_only)

    with conn:
        time_start = time.perf_counter()
        search_conn_exec(conn=conn, vector=vector, top_k=TOP_K)
        time_end = time.perf_counter()
        time_elapsed = (time_end - time_start) * 1000
        if not silent:
            print(f"[{timer_name}] Time taken: {time_elapsed:.2f} ms")


def single_query_with_existing_conn(
    vector: List[float],
    *,
    conn: "duckdb.DuckDBPyConnection",
    silent: bool = False,
    timer_name: Text = "SingleQueryWithExistingConn",
):
    time_start = time.perf_counter()
    search_conn_exec(conn=conn, vector=vector, top_k=TOP_K)
    time_end = time.perf_counter()
    time_elapsed = (time_end - time_start) * 1000
    if not silent:
        print(f"[{timer_name}] Time taken: {time_elapsed:.2f} ms")


def threading_query_with_new_conn(
    vector: List[float],
    *,
    threads_num: int,
    silent: bool = False,
    timer_name: Text = "ThreadingQueryWithNewConn",
):
    threads: List["threading.Thread"] = []
    for idx in range(threads_num):
        t = threading.Thread(
            target=single_query_with_new_conn,
            args=(vector,),
            kwargs={"silent": silent, "timer_name": f"{timer_name}-{idx}"},
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def threading_query_with_existing_conn(
    vector: List[float],
    *,
    conn: "duckdb.DuckDBPyConnection",
    threads_num: int,
    silent: bool = False,
    timer_name: Text = "ThreadingQueryWithExistingConn",
):
    threads: List["threading.Thread"] = []
    for idx in range(threads_num):
        t = threading.Thread(
            target=single_query_with_existing_conn,
            args=(vector,),
            kwargs={
                "conn": conn,
                "silent": silent,
                "timer_name": f"{timer_name}-{idx}",
            },
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def threading_pool_query_with_new_conn(
    vector: List[float],
    *,
    threads_num: int,
    silent: bool = False,
    timer_name: Text = "ThreadingPoolQueryWithNewConn",
):

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads_num) as executor:
        futures = [
            executor.submit(
                partial(
                    single_query_with_new_conn,
                    vector,
                    silent=silent,
                    timer_name=timer_name,
                ),
            )
            for _ in range(threads_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def threading_pool_query_with_existing_conn(
    vector: List[float],
    *,
    conn: "duckdb.DuckDBPyConnection",
    threads_num: int,
    silent: bool = False,
    timer_name: Text = "ThreadingPoolQueryWithExistingConn",
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads_num) as executor:
        futures = [
            executor.submit(
                partial(
                    single_query_with_existing_conn,
                    vector,
                    conn=conn,
                    silent=silent,
                    timer_name=timer_name,
                ),
            )
            for _ in range(threads_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def process_pool_query_with_new_conn(
    vector: List[float],
    *,
    processes_num: int,
    silent: bool = False,
    timer_name: Text = "ProcessPoolQueryWithNewConn",
):
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes_num) as executor:
        futures = [
            executor.submit(
                partial(
                    single_query_with_new_conn,
                    vector,
                    silent=silent,
                    timer_name=timer_name,
                    read_only=True,
                ),
            )
            for _ in range(processes_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def process_pool_query_with_existing_conn(
    vector: List[float],
    *,
    conn: "duckdb.DuckDBPyConnection",
    processes_num: int,
    silent: bool = False,
    timer_name: Text = "ProcessPoolQueryWithExistingConn",
):
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes_num) as executor:
        futures = [
            executor.submit(
                partial(
                    single_query_with_existing_conn,
                    vector,
                    conn=conn,
                    silent=silent,
                    timer_name=timer_name,
                ),
            )
            for _ in range(processes_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


async def async_to_thread_query_with_new_conn(
    vector: List[float],
    *,
    processes_num: int,
    silent: bool = False,
    timer_name: Text = "AsyncToThreadQueryWithNewConn",
):
    await asyncio.gather(
        *[
            asyncio.to_thread(
                single_query_with_new_conn,
                vector,
                silent=silent,
                timer_name=f"{timer_name}-{idx}",
            )
            for idx in range(processes_num)
        ]
    )


async def async_to_thread_query_with_existing_conn(
    vector: List[float],
    *,
    conn: "duckdb.DuckDBPyConnection",
    processes_num: int,
    silent: bool = False,
    timer_name: Text = "AsyncToThreadQueryWithExistingConn",
):
    await asyncio.gather(
        *[
            asyncio.to_thread(
                single_query_with_existing_conn,
                vector,
                conn=conn,
                silent=silent,
                timer_name=f"{timer_name}-{idx}",
            )
            for idx in range(processes_num)
        ]
    )


if __name__ == "__main__":
    vectors = get_vectors(
        ["Hello, world!", "How is AMD doing?", "What is the weather in Tokyo?"]
    )

    with get_conn(DB_FILEPATH, read_only=True) as conn:
        single_query_with_existing_conn(vectors[0], conn=conn, silent=True)  # Init conn
        single_query_with_existing_conn(vectors[1], conn=conn)
        threading_query_with_existing_conn(vectors[1], conn=conn, threads_num=3)
        threading_pool_query_with_existing_conn(
            vector=vectors[1], conn=conn, threads_num=3
        )
        # # Failed to run the same connection in different processes
        # process_pool_query_with_existing_conn(
        #     vector=vectors[1], conn=conn, processes_num=3
        # )
        asyncio.run(
            async_to_thread_query_with_existing_conn(
                vector=vectors[1], conn=conn, processes_num=3
            )
        )

    single_query_with_new_conn(vectors[0], silent=True)  # Init conn
    single_query_with_new_conn(vectors[1])
    threading_query_with_new_conn(vectors[1], threads_num=3)
    threading_pool_query_with_new_conn(vector=vectors[1], threads_num=3)
    process_pool_query_with_new_conn(vector=vectors[1], processes_num=3)
    asyncio.run(async_to_thread_query_with_new_conn(vector=vectors[1], processes_num=3))
