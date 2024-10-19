import json
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Text, Type, TypeVar, Union

import pymongo
from pydantic import BaseModel
from pymongo.synchronous.collection import Collection

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


def doc_exists(collection: "Collection", key: Text, value: Any) -> bool:
    """Check if a document exists in the collection.

    Parameters
    ----------
    collection : Collection
        The MongoDB collection to search within.
    key : Text
        The field name to check for the existence of the value.
    value : Any
        The value to search for in the specified field.

    Returns
    -------
    bool
        True if a document with the specified key and value exists, False otherwise.
    """

    return collection.count_documents({key: value}, limit=1) > 0


def docs_missing(
    collection: "Collection", key: Text, values: Sequence[Text]
) -> Set[Text]:
    """Identify which documents are missing from the collection.

    Parameters
    ----------
    collection : Collection
        The MongoDB collection to query.
    key : Text
        The field name to check for the existence of the values.
    values : Sequence[Text]
        A sequence of values to check against the collection.

    Returns
    -------
    Set[Text]
        A set of values that are not present in the collection.
    """

    values_set = {values} if isinstance(values, Text) else set(values)
    existing_ids = set(collection.distinct(key, {key: {"$in": list(values_set)}}))
    missing_ids = values_set - existing_ids
    return missing_ids


def fetch_docs_by_key_value(
    collection: "Collection",
    key: Text,
    value: Any,
    *,
    limit: int = 10,
    cast_to: Type[PydanticModel],
) -> List[PydanticModel]:
    """Fetch documents from the collection by a specific key-value pair.

    Parameters
    ----------
    collection : Collection
        The MongoDB collection to query.
    key : Text
        The field name to filter documents by.
    value : Any
        The value to match against the specified field.
    limit : int, optional
        The maximum number of documents to return (default is 10).
    cast_to : Type[PydanticModel]
        The Pydantic model to cast the retrieved documents to.

    Returns
    -------
    List[PydanticModel]
        A list of documents that match the specified key-value pair, cast to the provided Pydantic model.
    """  # noqa: E501

    docs = []
    for doc in collection.find({key: value}).limit(limit):
        docs.append(cast_to.model_validate(doc))
    return docs


def retrieve_doc_by_key_value(
    collection: "Collection",
    key: Text,
    value: Any,
    cast_to: Type[PydanticModel],
) -> Optional[PydanticModel]:
    """Retrieve a single document from the collection by a specific key-value pair.

    Parameters
    ----------
    collection : Collection
        The MongoDB collection to query.
    key : Text
        The field name to filter the document by.
    value : Any
        The value to match against the specified field.
    cast_to : Type[PydanticModel]
        The Pydantic model to cast the retrieved document to.

    Returns
    -------
    Optional[PydanticModel]
        The document that matches the specified key-value pair, cast to the provided Pydantic model, or None if no document is found.
    """  # noqa: E501

    doc = collection.find_one({key: value})
    return cast_to.model_validate(doc) if doc else None


def any_doc_to_mongo(doc: BaseModel | Dict, *, use_ts: bool = False) -> Dict:
    """Convert a Pydantic model or dictionary to a MongoDB-compatible dictionary.

    Parameters
    ----------
    doc : BaseModel | Dict
        The Pydantic model or dictionary to convert.
    use_ts : bool, optional
        If True, adds a timestamp to the document (default is False).

    Returns
    -------
    Dict
        A dictionary representation of the document suitable for MongoDB.
    """

    _doc = json.loads(doc.model_dump_json()) if isinstance(doc, BaseModel) else doc
    if use_ts:
        _doc["_ts"] = _doc.pop("created_at") or int(time.time())
    return _doc


def create_mongo_index(
    collection: "Collection",
    key: Text,
    *,
    expire_after_seconds: Optional[int] = None,
    sort: Union[int, Text] = pymongo.ASCENDING,
):
    """Create an index on a specified field in the collection.

    Parameters
    ----------
    collection : Collection
        The MongoDB collection to create the index on.
    key : Text
        The field name to index.
    expire_after_seconds : Optional[int], optional
        If provided, sets a TTL (time-to-live) for the index, specifying the number of seconds after which documents will expire (default is None).
    sort : Union[int, Text], optional
        The sort order for the index, either pymongo.ASCENDING or pymongo.DESCENDING (default is pymongo.ASCENDING).

    Returns
    -------
    None
    """  # noqa: E501

    if expire_after_seconds is None:
        collection.create_index([(key, sort)])
    else:
        collection.create_index([(key, sort)], expireAfterSeconds=expire_after_seconds)
