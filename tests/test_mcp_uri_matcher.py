import pytest

from mus.mcp.uri_matcher import URIMatcher

@pytest.fixture
def uri_matcher():
    matcher = URIMatcher()
    matcher.add_pattern("/users/{user_id}", "User")
    matcher.add_pattern("/posts/{post_id}/comments", "Comments")
    matcher.add_pattern("/static/resource", "StaticResource")
    matcher.add_pattern("/group/{group}/{id}", "Item")
    
    matcher.add_pattern("schema://{id}/resource", "NewResource")

    matcher.add_pattern("/trailer/{item_id}/", "TrailerItem")
    matcher.add_pattern("/trailer/", "TrailerRoot")


    return matcher

def test_uri_matcher(uri_matcher):
    # Test static pattern match
    match = uri_matcher.match("/static/resource")
    assert match is not None
    assert match.uri.uri == "/static/resource"
    assert match.values == {}

    # Test strip trailing slashes
    match = uri_matcher.match("/users/123/")
    assert match is not None
    assert match.uri.uri == "/users/{user_id}"
    assert match.values == {"user_id": "123"}
    match = uri_matcher.match("/static/resource/")
    assert match is not None
    assert match.uri.uri == "/static/resource"
    assert match.values == {}

    # Test matching patterns registered with trailing slashes
    match = uri_matcher.match("/trailer/item123")
    assert match is not None
    assert match.uri.uri == "/trailer/{item_id}"
    assert match.values == {"item_id": "item123"}
    
    match = uri_matcher.match("/trailer")
    assert match is not None
    assert match.uri.uri == "/trailer"
    assert match.values == {}

    # Test regex pattern match with parameters
    match = uri_matcher.match("/users/123")
    assert match is not None
    assert match.uri.uri == "/users/{user_id}"
    assert match.values == {"user_id": "123"}

    # Test another regex pattern with parameters
    match = uri_matcher.match("/posts/456/comments")
    assert match is not None
    assert match.uri.uri == "/posts/{post_id}/comments"
    assert match.values == {"post_id": "456"}

    # Test no match
    match = uri_matcher.match("/unknown/path")
    assert match is None

    # Test matching a pattern with multiple parameters
    match = uri_matcher.match("/group/admin/42")
    assert match is not None
    assert match.uri.uri == "/group/{group}/{id}"
    assert match.values == {"group": "admin", "id": "42"}

    # Test matching a schema URI
    match = uri_matcher.match("schema://123/resource")
    assert match is not None
    assert match.uri.uri == "schema://{id}/resource"
    assert match.values == {"id": "123"}

def test_uri_matcher_no_match(uri_matcher):
    # Test no match for a non-existent URI
    match = uri_matcher.match("/nonexistent/path")
    assert match is None

    # Test no match for a static pattern that doesn't exist
    match = uri_matcher.match("/static/unknown")
    assert match is None

def test_uri_matcher_add_pattern():
    matcher = URIMatcher()
    matcher.add_pattern("/new/resource", "NewResource")
    
    # Test matching the newly added static pattern
    match = matcher.match("/new/resource")
    assert match is not None
    assert match.uri.uri == "/new/resource"
    assert match.values == {}

    # Test matching a regex pattern
    matcher.add_pattern("/items/{item_id}", "Item")
    match = matcher.match("/items/789")
    assert match is not None
    assert match.uri.uri == "/items/{item_id}"
    assert match.values == {"item_id": "789"}

def test_uri_matcher_multiple_patterns():
    matcher = URIMatcher()
    matcher.add_pattern("/products/{product_id}", "Product")
    matcher.add_pattern("/categories/{category_id}/products", "CategoryProducts")

    # Test matching a product
    match = matcher.match("/products/42")
    assert match is not None
    assert match.uri.uri == "/products/{product_id}"
    assert match.values == {"product_id": "42"}

    # Test matching category products
    match = matcher.match("/categories/5/products")
    assert match is not None
    assert match.uri.uri == "/categories/{category_id}/products"
    assert match.values == {"category_id": "5"}

def test_uri_matcher_edge_cases():
    matcher = URIMatcher()
    matcher.add_pattern("/edge/case/{param}", "EdgeCase")

    # Test matching an edge case with special characters
    match = matcher.match("/edge/case/special@chars")
    assert match is not None
    assert match.uri.uri == "/edge/case/{param}"
    assert match.values == {"param": "special@chars"}

    # Test matching an empty string (should not match)
    match = matcher.match("")
    assert match is None

    
def test_uri_matcher_empty_patterns():
    matcher = URIMatcher()

    # Test matching an empty URI (should not match)
    match = matcher.match("")
    assert match is None

    # Test adding an empty pattern
    matcher.add_pattern("", "EmptyPattern")
    match = matcher.match("")
    assert match is not None
    assert match.uri.uri == ""
    assert match.values == {}

    # Test matching a non-empty URI after adding an empty pattern
    match = matcher.match("/some/path")
    assert match is None