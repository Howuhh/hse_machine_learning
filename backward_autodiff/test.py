from engine import Value

def test_grad():
    a, b = Value(-4.0), Value(2.0)

    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    
    assert round(g.value, 4) == 24.7041
    
    g.backward()
    
    assert round(a.grad, 4) == 138.8338
    assert round(b.grad, 4) == 645.5773

    print("Test Value: OK")    


if __name__ == "__main__":
    test_grad()