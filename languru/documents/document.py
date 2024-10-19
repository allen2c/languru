import hashlib
import os
import time
from itertools import zip_longest
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Text,
    Tuple,
    Type,
    Union,
    overload,
)

from cyksuid.v2 import ksuid
from diskcache import Cache
from openai import OpenAI
from pathvalidate import sanitize_filepath
from pydantic import BaseModel, ConfigDict, Field

from languru.documents._client import (
    DocumentQuerySet,
    DocumentQuerySetDescriptor,
    PointQuerySet,
    PointQuerySetDescriptor,
)
from languru.utils.openai_utils import embeddings_create_with_cache

if TYPE_CHECKING:
    import duckdb


class Point(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    TABLE_NAME: ClassVar[Text] = "points"
    EMBEDDING_MODEL: ClassVar[Text] = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: ClassVar[int] = 512
    EMBEDDING_CACHE_PATH: ClassVar[Text] = os.path.join(
        os.path.expanduser("~"), ".languru"
    )
    EMBEDDING_CACHE_LIMIT: ClassVar[int] = 2**30  # 1GB
    objects: ClassVar["PointQuerySetDescriptor"] = PointQuerySetDescriptor()

    point_id: Text = Field(
        default_factory=lambda: f"pt_{str(ksuid())}",
        description="The unique and primary ID of the point.",
    )
    document_id: Text = Field(
        description="The ID of the document that the point belongs to."
    )
    content_md5: Text = Field(
        description="The MD5 hash of the document that the point belongs to."
    )
    embedding: List[float] = Field(
        max_length=512,
        description="The embedding of the point.",
        default_factory=list,
    )

    @classmethod
    def query_set(cls) -> "PointQuerySet":
        from languru.documents._client import PointQuerySet

        return PointQuerySet(cls, model_with_score=PointWithScore)

    @classmethod
    def embedding_cache(cls, model: Text) -> Cache:
        cache_path = sanitize_filepath(os.path.join(cls.EMBEDDING_CACHE_PATH, model))
        return Cache(directory=cache_path, size_limit=cls.EMBEDDING_CACHE_LIMIT)

    def is_embedded(self) -> bool:
        return False if len(self.embedding) == 0 else True


class PointWithScore(Point):
    # model_config = ConfigDict(str_strip_whitespace=True, extra="allow")

    relevance_score: float = Field(description="The score of the point.")


class Document(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    TABLE_NAME: ClassVar[Text] = "documents"
    POINT_TYPE: ClassVar[Type[Point]] = Point
    objects: ClassVar["DocumentQuerySetDescriptor"] = DocumentQuerySetDescriptor()

    document_id: Text = Field(
        default_factory=lambda: f"doc_{str(ksuid())}",
        description="The unique and primary ID of the document.",
    )
    name: Text = Field(max_length=255, description="The unique name of the document.")
    content: Text = Field(max_length=5000, description="The content of the document.")
    content_md5: Text = Field(description="The MD5 hash of the document content.")
    metadata: Dict[Text, Any] = Field(
        default_factory=dict, description="Any additional metadata for the document."
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The timestamp of when the document was created.",
    )
    updated_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The timestamp of when the document was last updated.",
    )

    @classmethod
    def query_set(cls) -> "DocumentQuerySet":
        from languru.documents._client import DocumentQuerySet

        return DocumentQuerySet(cls)

    @classmethod
    def from_content(
        cls, name: Text, content: Text, *, metadata: Optional[Dict[Text, Any]] = None
    ) -> "Document":
        return cls.model_validate(
            {
                "content": content,
                "content_md5": cls.hash_content(content),
                "name": name,
                "metadata": metadata or {},
            }
        )

    @classmethod
    def hash_content(cls, content: Text) -> Text:
        return hashlib.md5(content.strip().encode("utf-8")).hexdigest()

    @overload
    def to_points(
        self,
        *,
        openai_client: Optional["OpenAI"] = None,
        embeddings: Optional[List[List[float]]] = None,
        with_document_card: Literal[False] = False,
        debug: bool = False,
    ) -> List["Point"]: ...

    @overload
    def to_points(
        self,
        *,
        openai_client: Optional["OpenAI"] = None,
        embeddings: Optional[List[List[float]]] = None,
        with_document_card: Literal[True] = True,
        debug: bool = False,
    ) -> List[Tuple["Point", Text]]: ...

    def to_points(
        self,
        *,
        openai_client: Optional["OpenAI"] = None,
        embeddings: Optional[List[List[float]]] = None,
        with_document_card: bool = False,
        debug: bool = False,
    ) -> Union[List["Point"], List[Tuple["Point", Text]]]:
        """
        Convert the document into a list of Point objects, optionally with embeddings.

        This method creates Point objects from the document, which can be used for
        vector search and other operations. It can also generate embeddings for the
        document content if an OpenAI client is provided.

        Parameters
        ----------
        openai_client : Optional[OpenAI], optional
            An instance of the OpenAI client to use for generating embeddings.
            If not provided, embeddings will not be generated unless passed explicitly.
        embeddings : Optional[List[List[float]]], optional
            Pre-computed embeddings for the document. If provided, these will be used
            instead of generating new embeddings.
        with_document_card : bool, default False
            If True, returns a tuple of (Point, document_card) for each point.
            If False, returns only the Point objects.
        debug : bool, default False
            If True, enables debug mode for additional logging or verbose output.

        Returns
        -------
        Union[List[Point], List[Tuple[Point, Text]]]
            A list of Point objects if with_document_card is False,
            or a list of tuples (Point, document_card) if with_document_card is True.

        Raises
        ------
        ValueError
            If the number of provided embeddings does not match the number of document cards.

        Notes
        -----
        - The method first strips the document content.
        - If embeddings are not provided and an OpenAI client is given, it generates
        embeddings using the specified embedding model.
        - The method uses the POINT_TYPE class variable to determine the Point class to use.
        - The embedding model and dimensions are determined by class variables
        EMBEDDING_MODEL and EMBEDDING_DIMENSIONS.

        See Also
        --------
        Point : The class used to create point objects.
        embeddings_create_with_cache : Function used to generate embeddings with caching.
        """  # noqa: E501

        self.strip()

        out = []
        base_params = MappingProxyType(
            {"document_id": self.document_id, "content_md5": self.content_md5}
        )

        doc_cards = self.to_document_cards()

        # Create embeddings if not provided
        if embeddings is None and openai_client is not None:
            embeddings = embeddings_create_with_cache(
                input=doc_cards,
                model=self.POINT_TYPE.EMBEDDING_MODEL,
                dimensions=self.POINT_TYPE.EMBEDDING_DIMENSIONS,
                openai_client=openai_client,
                cache=self.POINT_TYPE.embedding_cache(self.POINT_TYPE.EMBEDDING_MODEL),
            )

        # Validate embeddings
        if embeddings is not None:
            if len(embeddings) != len(doc_cards):
                raise ValueError(
                    "The number of provided embeddings must match "
                    + "the number of document cards."
                )

        # Create points
        for embedding, doc_card in zip_longest(
            embeddings or [], doc_cards, fillvalue=None
        ):
            pt_params: Dict = dict(base_params)
            if embedding is not None:
                pt_params["embedding"] = embedding
            if with_document_card:
                out.append((self.POINT_TYPE.model_validate(pt_params), doc_card))
            else:
                out.append(self.POINT_TYPE.model_validate(pt_params))

        return out

    def has_points(
        self, *, conn: "duckdb.DuckDBPyConnection", debug: bool = False
    ) -> bool:
        """
        Check if the document has any associated points in the database.

        This method queries the database to determine if there are any Point objects
        associated with the current document.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            A connection to the DuckDB database.
        debug : bool, default False
            If True, enables debug mode for additional logging or verbose output.

        Returns
        -------
        bool
            True if the document has at least one associated point, False otherwise.

        Notes
        -----
        - This method uses the POINT_TYPE class variable to determine which Point
        class to use for the query.
        - The query is performed using the `count` method of the Point objects manager.

        See Also
        --------
        Point : The class representing points in the database.
        PointQuerySet : The query set used for database operations on Point objects.
        """

        return (
            self.POINT_TYPE.objects.count(
                document_id=self.document_id, conn=conn, debug=debug
            )
            > 0
        )

    def are_points_current(
        self, conn: "duckdb.DuckDBPyConnection", *, debug: bool = False
    ) -> bool:
        """
        Check if all points associated with the document are up-to-date.

        This method verifies that all Point objects associated with the document
        have the same content_md5 as the current document, indicating that they
        are synchronized with the latest version of the document.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            A connection to the DuckDB database.
        debug : bool, default False
            If True, enables debug mode for additional logging or verbose output.

        Returns
        -------
        bool
            True if all associated points are up-to-date, False otherwise.

        Notes
        -----
        - The method uses pagination to handle potentially large numbers of points.
        - It returns False immediately if no points are found for the document.
        - The check is performed by comparing the content_md5 of each point with
        the current document's content_md5.

        See Also
        --------
        Point : The class representing points in the database.
        PointQuerySet : The query set used for database operations on Point objects.
        """

        after = None
        has_more = True
        while has_more:
            page_points = Point.objects.list(
                document_id=self.document_id, after=after, conn=conn, debug=debug
            )

            # Return False if no points are found
            if len(page_points.data) == 0 and after is None:
                return False

            # Return False if any point is out of date
            for pt in page_points.data:
                if pt.content_md5 != self.content_md5:
                    return False

            # Update pagination state
            after = page_points.last_id
            has_more = page_points.has_more

        return True

    def sync_points(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        openai_client: "OpenAI",
        with_embeddings: bool = False,
        force: bool = False,
        debug: bool = False,
    ) -> Tuple["Point", ...]:
        """
        Synchronize the document's points in the database.

        This method ensures that the points associated with the document in the
        database are up-to-date. It can create new points, update existing ones,
        and optionally generate embeddings.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection
            A connection to the DuckDB database.
        openai_client : OpenAI
            An instance of the OpenAI client to use for generating embeddings.
        with_embeddings : bool, default False
            If True, generates embeddings for the points.
        force : bool, default False
            If True, forces synchronization even if points are already up-to-date.
        debug : bool, default False
            If True, enables debug mode for additional logging or verbose output.

        Returns
        -------
        Tuple[Point, ...]
            A tuple containing the synchronized Point objects.

        Raises
        ------
        ValueError
            If no points are created during the synchronization process.

        Notes
        -----
        - This method uses the `documents_sync_points` method of the document's
        query set to perform the synchronization.
        - The method is designed to work with a single document, but internally
        uses a method that can handle multiple documents.

        See Also
        --------
        DocumentQuerySet.documents_sync_points : Method used for synchronizing points.
        Point : The class representing points in the database.
        """

        docs_pts = self.__class__.objects.documents_sync_points(
            [self],
            conn=conn,
            openai_client=openai_client,
            with_embeddings=with_embeddings,
            force=force,
            debug=debug,
        )
        if len(docs_pts) == 0:
            raise ValueError("No points created")

        return docs_pts[0]

    def to_document_cards(self, *args, **kwargs) -> List[Text]:
        """
        Convert the document content into a list of document cards.

        This method prepares the document content for embedding or other processing
        by converting it into a list of text strings (cards).

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        List[Text]
            A list containing a single string, which is the stripped content of the document.

        Notes
        -----
        - In the base implementation, this method simply returns a list with one item:
        the stripped content of the document.
        - Subclasses may override this method to implement more complex document
        card creation logic.
        """  # noqa: E501

        return [self.content.strip()]

    @classmethod
    def to_query_cards(cls, query: Text, *args, **kwargs) -> List[Text]:
        """
        Convert a query string into a list of query cards.

        This class method prepares a query string for embedding or other processing
        by converting it into a list of text strings (cards).

        Parameters
        ----------
        query : Text
            The query string to be converted into cards.
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        List[Text]
            A list containing a single string, which is the stripped query.

        Notes
        -----
        - In the base implementation, this method simply returns a list with one item:
        the stripped query string.
        - Subclasses may override this method to implement more complex query
        card creation logic.
        """

        return [query.strip()]

    def strip(self, *, copy: bool = False) -> "Document":
        """
        Strip whitespace from the document's content and update the content_md5.

        This method removes leading and trailing whitespace from the document's content.
        If the content changes, it updates the content_md5 and the updated_at timestamp.

        Parameters
        ----------
        copy : bool, default False
            If True, creates a deep copy of the document before modifying it.
            If False, modifies the document in-place.

        Returns
        -------
        Document
            The document with stripped content, either the original (modified in-place)
            or a new copy.

        Notes
        -----
        - This method uses the `hash_content` class method to generate the new MD5 hash.
        - If the content_md5 changes, the updated_at timestamp is set to the current time.

        See Also
        --------
        Document.hash_content : Class method used to generate the MD5 hash of the content.
        """  # noqa: E501

        _doc = self.model_copy(deep=True) if copy else self
        _doc.content = _doc.content.strip()
        new_md5 = self.hash_content(_doc.content)
        if _doc.content_md5 != new_md5:
            _doc.content_md5 = new_md5
            _doc.updated_at = int(time.time())
        return _doc


class SearchResult(BaseModel):
    query: Optional[Text] = Field(
        default=None, description="The query that was used for searching."
    )
    matches: List[PointWithScore] = Field(
        default_factory=list, description="The points that match the search query."
    )
    documents: List[Document] = Field(
        default_factory=list, description="The documents that match the search query."
    )
    total_results: int = Field(
        default=0, description="The total number of results found."
    )
    execution_time: float = Field(
        default=0.0,
        description="The time taken to execute the search query in seconds.",
    )
    relevance_score: Optional[float] = Field(
        default=None, description="An overall relevance score for the search result."
    )
    highlight: Optional[Dict[Text, List[Text]]] = Field(
        default=None,
        description="Highlighted snippets of text that match the search query.",
    )
    facets: Optional[Dict[Text, Dict[Text, int]]] = Field(
        default=None,
        description="Faceted search results for categorization.",
    )
    suggestions: Optional[List[Text]] = Field(
        default=None, description="Suggested search terms related to the query."
    )
