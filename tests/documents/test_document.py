import copy
from itertools import chain
from pprint import pformat
from typing import List

import duckdb
import pytest
from openai import OpenAI

from languru.documents.document import Document, Point
from languru.exceptions import NotFound, NotSupported
from languru.openai_plugins.clients.voyage import VoyageOpenAI

raw_docs = [
    {
        "name": "AMD's New AI Chips",
        "content": "AMD unveiled its latest AI chips, including the Instinct MI325X accelerator and 5th Gen EPYC processors, at its Advancing AI 2024 event, aiming to challenge Nvidia's dominance in the lucrative data center GPU market and compete with Intel in server CPUs. According to reports from Yahoo Finance and PYMNTS, while AMD's new offerings show improved performance, analysts suggest they still lag behind Nvidia's upcoming Blackwell chips by about a year.",  # noqa: E501
        "metadata": {"source": "Yahoo Finance"},
    },
    {
        "name": "Jupiter's 190 Year Old Storm",
        "content": "Jupiter's Great Red Spot, a colossal storm that has fascinated astronomers for centuries, is undergoing unexpected changes. Recent observations by the Hubble Space Telescope reveal that this 190-year-old vortex is oscillating in size and shape, behaving like a cosmic stress ball as it continues to shrink and evolve.",  # noqa: E501
    },
    {
        "name": "Tesla Unveiled Cybercab",
        "content": 'Tesla has unveiled its highly anticipated Cybercab robotaxi, showcasing a fleet of 20 sleek, autonomous vehicles at the "We, Robot" event held at Warner Bros. Discovery studio. As reported by TechCrunch, the Cybercab features a design reminiscent of a smaller Cybertruck, complete with suicide doors and no steering wheel or pedals, signaling Tesla\'s bold step towards a fully autonomous future.',  # noqa: E501
    },
]

openai_client = OpenAI()
vo_client = VoyageOpenAI()


def test_document_operations():
    conn: "duckdb.DuckDBPyConnection" = duckdb.connect(":memory:")

    docs = _create_docs(conn)

    # Get documents
    retrieved_docs = Document.objects.retrieve(
        docs[0].document_id, conn=conn, debug=True
    )
    assert retrieved_docs is not None
    assert retrieved_docs.document_id == docs[0].document_id
    after = None
    has_more = True
    docs_ids_set = set()
    while has_more:
        page_docs = Document.objects.list(after=after, conn=conn, debug=True, limit=1)
        after = page_docs.last_id
        has_more = len(page_docs.data) > 0
        docs_ids_set.update(doc.document_id for doc in page_docs.data)
    assert docs_ids_set == set(doc.document_id for doc in docs)

    # Count documents
    assert Document.objects.count(conn=conn, debug=True) == len(docs)

    # Update document
    new_doc_name = "[Updated] " + docs[0].name
    updated_doc = Document.objects.update(
        docs[0].document_id,
        name=new_doc_name,
        metadata={"written_by": "Languru"},
        conn=conn,
        debug=True,
    )
    new_meta = copy.deepcopy(docs[0].metadata)
    new_meta["written_by"] = "Languru"  # Merge strategy is update extra fields
    assert updated_doc.name == new_doc_name
    assert pformat(updated_doc.metadata) == pformat(new_meta)

    # Remove document
    Document.objects.remove(docs[0].document_id, conn=conn, debug=True)
    with pytest.raises(NotFound):
        Document.objects.retrieve(docs[0].document_id, conn=conn, debug=True)

    # Create document  again
    docs[0] = Document.objects.create(
        Document.from_content(**raw_docs[0]), conn=conn, debug=True
    )


def test_point_operations():
    conn: "duckdb.DuckDBPyConnection" = duckdb.connect(":memory:")

    docs = _create_docs(conn)
    points = _create_points(docs, conn)

    # Get points
    retrieved_points = Point.objects.retrieve(points[0].point_id, conn=conn, debug=True)
    assert retrieved_points is not None
    assert retrieved_points.point_id == points[0].point_id
    assert retrieved_points.is_embedded() is False
    assert (
        Point.objects.retrieve(
            points[0].point_id, conn=conn, debug=True, with_embedding=True
        ).is_embedded()
        is True
    )

    # List points
    after = None
    has_more = True
    points_ids_set = set()
    while has_more:
        page_points = Point.objects.list(after=after, conn=conn, debug=True, limit=1)
        after = page_points.last_id
        has_more = len(page_points.data) > 0
        points_ids_set.update(pt.point_id for pt in page_points.data)
    assert points_ids_set == set(pt.point_id for pt in points)
    page_points = Point.objects.list(
        content_md5=points[0].content_md5, conn=conn, debug=True
    )
    assert len(page_points.data) == 1
    assert page_points.data[0].point_id == points[0].point_id

    # Count points
    assert (
        Point.objects.count(document_id=docs[0].document_id, conn=conn, debug=True) == 1
    )

    # Update points is not supported
    with pytest.raises(NotSupported):
        Point.objects.update(points[0].point_id, conn=conn, debug=True)

    # Remove points
    Point.objects.remove(points[0].point_id, conn=conn, debug=True)
    with pytest.raises(NotFound):
        Point.objects.retrieve(points[0].point_id, conn=conn, debug=True)


def test_document_search():
    conn: "duckdb.DuckDBPyConnection" = duckdb.connect(":memory:")

    docs = _create_docs(conn)
    _create_points(docs, conn)

    # Check if documents have points and points are current
    for _doc in docs:
        assert _doc.has_points(conn=conn, debug=True) is True
        assert _doc.are_points_current(conn=conn, debug=True) is True

    # Search
    search_results = Document.objects.search(
        "Jupiter's 190 Year Old Storm",
        conn=conn,
        openai_client=openai_client,
        with_embedding=True,
        with_documents=True,
        top_k=1,
        debug=True,
    )
    assert search_results.matches
    assert search_results.documents
    assert len(search_results.matches) == 1
    assert search_results.matches[0].relevance_score > 0
    assert len(search_results.matches[0].embedding) == Point.EMBEDDING_DIMENSIONS
    assert len(search_results.documents) == 1
    assert (
        search_results.matches[0].document_id == search_results.documents[0].document_id
    )
    assert (
        search_results.matches[0].content_md5 == search_results.documents[0].content_md5
    )


def test_documents_bulk_create():
    conn: "duckdb.DuckDBPyConnection" = duckdb.connect(":memory:")
    Document.objects.touch(conn=conn, debug=True)

    docs = Document.objects.bulk_create(
        [Document.from_content(**_raw_doc) for _raw_doc in raw_docs],
        conn=conn,
        debug=True,
    )
    docs_pts = Document.objects.documents_to_points(
        docs, openai_client=openai_client, debug=True
    )
    _all_pts = list(chain(*docs_pts))
    Point.objects.bulk_create(_all_pts, conn=conn, debug=True)

    assert Point.objects.count(conn=conn, debug=True) == len(_all_pts)

    # Search
    search_results = Document.objects.search(
        "Jupiter's 190 Year Old Storm",
        conn=conn,
        openai_client=openai_client,
        with_documents=True,
        debug=True,
    )
    assert search_results.matches
    assert search_results.documents


def test_documents_sync_points():
    conn: "duckdb.DuckDBPyConnection" = duckdb.connect(":memory:")
    Document.objects.touch(conn=conn, debug=True)

    _batch_docs = Document.objects.bulk_create(
        [Document.from_content(**_raw_doc) for _raw_doc in raw_docs],
        conn=conn,
        debug=True,
    )
    Document.objects.documents_sync_points(
        _batch_docs,
        conn=conn,
        openai_client=openai_client,
        debug=True,
    )
    Document.objects.documents_sync_points(
        _batch_docs,
        conn=conn,
        openai_client=openai_client,
        debug=True,
    )
    search_results = Document.objects.search(
        "Jupiter's 190 Year Old Storm",
        conn=conn,
        openai_client=openai_client,
        rerank_client=vo_client,
        with_documents=True,
        debug=True,
    )
    assert len(search_results.matches) == len(raw_docs)
    assert len(search_results.documents) == len(raw_docs)
    for _m, _d in zip(search_results.matches, search_results.documents):
        assert _m.document_id == _d.document_id
        assert _m.content_md5 == _d.content_md5
    assert all(
        pt_1.relevance_score >= pt_2.relevance_score
        for pt_1, pt_2 in zip(search_results.matches, search_results.matches[1:])
    )


def _create_docs(conn: "duckdb.DuckDBPyConnection") -> List["Document"]:
    # Touch table
    Document.objects.touch(conn=conn, debug=True)

    # Create documents
    docs: List["Document"] = []
    for _raw_doc in raw_docs:
        doc = Document.objects.create(
            Document.from_content(**_raw_doc), conn=conn, debug=True
        )
        docs.append(doc)
    return docs


def _create_points(
    docs: List["Document"], conn: "duckdb.DuckDBPyConnection"
) -> List["Point"]:
    # Create points
    points: List["Point"] = []
    for _doc in docs:
        for _pt in _doc.to_points(openai_client=openai_client):
            points.append(_pt)
            Point.objects.create(_pt, conn=conn, debug=True)
    return points
