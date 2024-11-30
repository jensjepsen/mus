

def patch_anthropic():
    from anthropic.lib.bedrock._auth import _get_session
    from unittest import mock
    import httpx

    def get_auth_headers(
        *,
        method: str,
        url: str,
        headers: httpx.Headers,
        aws_access_key: str | None,
        aws_secret_key: str | None,
        aws_session_token: str | None,
        region: str | None,
        profile: str | None,
        data: str | None,
    ) -> dict[str, str]:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        session = _get_session(
            profile=profile,
            region=region,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )

        # The connection header may be stripped by a proxy somewhere, so the receiver
        # of this message may not see this header, so we remove it from the set of headers
        # that are signed.
        headers = headers.copy()
        del headers["connection"]
        del headers["content-length"]
        del headers["host"]
        del headers["accept-encoding"]

        request = AWSRequest(method=method.upper(), url=url, headers=headers, data=data)
        credentials = session.get_credentials()
        if not credentials:
            raise RuntimeError("could not resolve credentials from session")

        signer = SigV4Auth(credentials, "bedrock", session.region_name)
        signer.add_auth(request)

        prepped = request.prepare()

        return {key: value for key, value in dict(prepped.headers).items() if value is not None}

    from partial_json_parser import loads as _loads

    def loads(data: str, *args, **kwargs):
        kwargs.pop("partial_mode", None)
        return _loads(data.decode("utf-8"), *args, **kwargs)
    import sys
    sys.modules["jiter"] = mock.Mock()
    import jiter
    setattr(jiter, "from_json", loads)


    p = mock.patch("anthropic.lib.bedrock._auth.get_auth_headers", get_auth_headers)
    p.__enter__()