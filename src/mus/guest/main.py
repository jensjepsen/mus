from mus import Mus
import extism

@extism.plugin_fn
def mus() -> Mus:
    return Mus()