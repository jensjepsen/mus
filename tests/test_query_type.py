from mus.llm.types import Query, File, Assistant

def test_query_text_plus_file():
    q = "Hello" + File(b64type="image/png", content="sadf")
    assert isinstance(q, Query)
    assert len(q.val) == 2
    assert isinstance(q.val[0], str)
    assert isinstance(q.val[1], File)
    assert q.val[0] == "Hello"
    assert q.val[1].content == "sadf"

def test_query_dedent():
    text = """\
    Hello, this is a test.
    Of some things."""
    q = Query(text)
    assert isinstance(q, Query)
    assert len(q.val) == 1
    assert isinstance(q.val[0], str)
    assert q.val[0] == "Hello, this is a test.\nOf some things."

    text2 = """\
        Hello, again, this is a test.
        Of some other things."""
    q2 = Query(text2)
    assert isinstance(q2, Query)
    assert len(q2.val) == 1
    assert isinstance(q2.val[0], str)
    assert q2.val[0] == "Hello, again, this is a test.\nOf some other things."

def test_query_text_plus_assistant():
    q = "Hello" + Assistant("I'm an assistant")
    assert isinstance(q, Query)
    assert len(q.val) == 2
    assert isinstance(q.val[0], str)
    assert isinstance(q.val[1], Assistant)
    assert q.val[0] == "Hello"
    assert q.val[1].val == "I'm an assistant"
    
def test_query_query_plus_text():
    q1 = Query(["Initial query"])
    q2 = q1 + "Additional text"
    assert isinstance(q2, Query)
    assert len(q2.val) == 2
    assert q2.val[0] == "Initial query"
    assert q2.val[1] == "Additional text"

def test_query_query_plus_file():
    q1 = Query(["Initial query"])
    f = File(b64type="image/png", content="image_content")
    q2 = q1 + f
    assert isinstance(q2, Query)
    assert len(q2.val) == 2
    assert q2.val[0] == "Initial query"
    assert isinstance(q2.val[1], File)
    assert q2.val[1].content == "image_content"

def test_query_query_plus_query():
    q1 = Query(["Query 1"])
    q2 = Query(["Query 2"])
    q3 = q1 + q2
    assert isinstance(q3, Query)
    assert len(q3.val) == 2
    assert q3.val[0] == "Query 1"
    assert q3.val[1] == "Query 2"

def test_query_parse_query():
    # Test parsing a string
    q1 = Query.parse("Hello")
    assert isinstance(q1, Query)
    assert len(q1.val) == 1
    assert q1.val[0] == "Hello"

    # Test parsing a File
    f = File(b64type="image/png", content="image_content")
    q2 = Query.parse(f)
    assert isinstance(q2, Query)
    assert len(q2.val) == 1
    assert isinstance(q2.val[0], File)
    assert q2.val[0].content == "image_content"

    # Test parsing an Assistant
    a = Assistant("Assistant message")
    q3 = Query.parse(a)
    assert isinstance(q3, Query)
    assert len(q3.val) == 1
    assert isinstance(q3.val[0], Assistant)
    assert q3.val[0].val == "Assistant message"

    # Test parsing a list
    q4 = Query.parse(["Hello", f, a])
    assert isinstance(q4, Query)
    assert len(q4.val) == 3
    assert q4.val[0] == "Hello"
    assert isinstance(q4.val[1], File)
    assert isinstance(q4.val[2], Assistant)

    # Test parsing a Query
    q5 = Query(["Existing query"])
    q6 = Query.parse(q5)
    assert isinstance(q6, Query)
    assert len(q6.val) == 1
    assert q6.val[0] == "Existing query"

def test_query_to_deltas():
    q = Query(["Test query"])
    deltas = q.to_deltas()
    assert isinstance(deltas, list)
    assert len(deltas) == 1
    assert deltas[0] == q

def test_query_assistant_plus_assistant():
    a1 = Assistant("Assistant 1")
    a2 = Assistant("Assistant 2")
    result = a1 + a2
    assert isinstance(result, Assistant)
    assert result.val == "Assistant 1Assistant 2"

def test_query_assistant_plus_query():
    a = Assistant("Assistant")
    q = Query(["Query"])
    result = a + q
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert isinstance(result.val[0], Assistant)
    assert result.val[0].val == "Assistant"
    assert result.val[1] == "Query"

def test_query_query_plus_assistant():
    q = Query(["Query"])
    a = Assistant("Assistant")
    result = q + a
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert result.val[0] == "Query"
    assert isinstance(result.val[1], Assistant)
    assert result.val[1].val == "Assistant"

def test_query_assistant_plus_string():
    a = Assistant("Assistant")
    s = "String"
    result = a + s
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert isinstance(result.val[0], Assistant)
    assert result.val[0].val == "Assistant"
    assert result.val[1] == "String"

def test_query_string_plus_assistant():
    s = "String"
    a = Assistant("Assistant")
    result = s + a
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert result.val[0] == "String"
    assert isinstance(result.val[1], Assistant)
    assert result.val[1].val == "Assistant"

def test_query_assistant_plus_file():
    a = Assistant("Assistant")
    f = File(b64type="image/png", content="image_content")
    result = a + f
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert isinstance(result.val[0], Assistant)
    assert result.val[0].val == "Assistant"
    assert isinstance(result.val[1], File)
    assert result.val[1].content == "image_content"

def test_query_file_plus_assistant():
    f = File(b64type="image/png", content="image_content")
    a = Assistant("Assistant")
    result = f + a
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert isinstance(result.val[0], File)
    assert result.val[0].content == "image_content"
    assert isinstance(result.val[1], Assistant)
    assert result.val[1].val == "Assistant"

def test_query_assistant_initialization():
    a = Assistant("Hello, I'm an assistant")
    assert isinstance(a, Assistant)
    assert a.val == "Hello, I'm an assistant"

def test_query_assistant_plus_assistant_multiple():
    a1 = Assistant("Assistant 1")
    a2 = Assistant("Assistant 2")
    a3 = Assistant("Assistant 3")
    result = a1 + a2 + a3
    assert isinstance(result, Assistant)
    assert result.val == "Assistant 1Assistant 2Assistant 3"

def test_query_assistant_radd_with_string():
    a = Assistant("Assistant")
    result = "Prefix " + a
    assert isinstance(result, Query)
    assert len(result.val) == 2
    assert result.val[0] == "Prefix "
    assert isinstance(result.val[1], Assistant)
    assert result.val[1].val == "Assistant"

def test_query_assistant_radd_with_assistant():
    a1 = Assistant("Assistant 1")
    a2 = Assistant("Assistant 2")
    result = a1 + a2
    assert isinstance(result, Assistant)
    assert result.val == "Assistant 1Assistant 2"