from ...utils import is_module_available

if is_module_available("pyximport"):
    import pyximport

    pyximport.install()
    try:
        from gliner.decoding.trie.labels_trie import LabelsTrie
    except ImportError:
        from .python_labels_trie import LabelsTrie
else:
    from .python_labels_trie import LabelsTrie
