import six


def flatten(struct):
    """
    Creates a flat list of all all items in structured output (dicts, lists, items):
    .. code-block:: python
        >>> sorted(flatten({'a': 'foo', 'b': 'bar'}))
        ['bar', 'foo']
        >>> sorted(flatten(['foo', ['bar', 'troll']]))
        ['bar', 'foo', 'troll']
        >>> flatten('foo')
        ['foo']
        >>> flatten(42)
        [42]
    """
    if struct is None:
        return []
    flat = []
    if isinstance(struct, dict):
        for _, result in six.iteritems(struct):
            flat += flatten(result)
        return flat
    if isinstance(struct, six.string_types):
        return [struct]

    try:
        # if iterable
        iterator = iter(struct)
    except TypeError:
        return [struct]

    for result in iterator:
        flat += flatten(result)
    return flat