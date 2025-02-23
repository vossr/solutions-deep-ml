class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self, ), 'ReLU')

        def _backward():
            if self.data > 0: self.grad += 1 * out.grad
            else: self.grad += 0
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        self.grad = 1
        build_topo(self)
        for v in reversed(topo):
            v._backward()

a = Value(2)
b = Value(-3)
c = Value(10)
d = a + b * c
e = d.relu()
e.backward()
print(a, b, c, d, e)
