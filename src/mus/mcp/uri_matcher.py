import typing as t
import re
import dataclasses as dc
T = t.TypeVar('T')

find_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

@dc.dataclass(frozen=True)
class URIMatch(t.Generic[T]):
    """
    A dataclass representing a match for a URI pattern.
    
    :param pattern: The URI pattern that was matched.
    :param value: The value associated with the matched pattern.
    """
    uri: "URI[T]"
    values: t.Dict[str, str]

    def __str__(self) -> str:
        return f"URIMatch(uri={self.uri}, values={self.values})"
    
@dc.dataclass(frozen=True)
class URI(t.Generic[T]):
    """
    A dataclass representing a URI with a value.

    :param uri: The URI string.
    :param value: The value associated with the URI.
    """
    uri: str
    value: T

    def __str__(self) -> str:
        return f"URI(uri={self.uri}, value={self.value})"

class URIMatcher[T]():
    """
    A class to match URIs against a set of patterns.
    """

    def __init__(self):
        """
        Initialize the URIMatcher with a list of patterns.

        :param patterns: List of URI patterns to match against.
        """
        self.patterns : dict[re.Pattern, URI[T]] = {}
        self.static : dict[str, URI[T]] = {}

    def add_pattern(self, pattern: str, value: T) -> t.Literal["static", "dynamic"]:
        """
        Add a pattern to the matcher.

        :param pattern: The URI pattern to match.
        :param value: The value associated with the pattern.
        """
        pattern = pattern[:-1] if pattern.endswith("/") else pattern  # Remove trailing slash for consistency

        # is it a static pattern?
        if not find_pattern.search(pattern):
            # static pattern, store it directly
            self.static[pattern] = URI(
                uri=pattern,
                value=value
            )
            return "static"
        else:
            regex_pattern = find_pattern.sub(r'(?P<\1>[^/]+)', pattern)
            compiled_pattern = re.compile(regex_pattern)
            self.patterns[compiled_pattern] = URI(
                uri=pattern,
                value=value
            )
            return "dynamic"

    def match(self, uri: str) -> t.Optional[URIMatch[T]]:
        """
        Match a URI against the stored patterns.

        :param uri: The URI to match.
        :return: The value associated with the matched pattern, or None if no match is found.
        """
        uri = uri[:-1] if uri.endswith("/") else uri  # Remove trailing slash for consistency
        # Check static patterns first
        if uri in self.static:
            return URIMatch(
                uri=self.static[uri],
                values={}
            )

        # Check regex patterns
        for pattern, _uri in self.patterns.items():
            if match := re.match(pattern, uri):
                return URIMatch(
                    uri=_uri,
                    values=match.groupdict()
                )

        return None