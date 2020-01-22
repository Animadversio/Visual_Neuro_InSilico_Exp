from multiprocessing import Value


class DynamicParameter:
    """
    shallow wrapper for multiprocessing.Value
    defines a struct around it:
        type: converter/typecast function
        description: short str describing the value
        value: value extracted from the synchronized Value
    """
    def __init__(self, typecode, value, description=''):
        # https://docs.python.org/3.6/library/array.html#module-array
        if typecode == 'd' or typecode == 'f':
            self._type = float
        elif typecode == 'i':
            self._type = int
        else:
            raise ValueError('typecode %s not understood' % typecode)
        self._synchronized_value = Value(typecode, value)
        self.description = description

    def set_value(self, value):
        # if value mismatch, will throw ValueError
        value = self._type(value)
        # if type mismatch, will throw TypeError; but shouldn't because we have just converted
        self._synchronized_value.value = value

    @property
    def value(self):
        return self._synchronized_value.value

    @property
    def type(self):
        return self._type
