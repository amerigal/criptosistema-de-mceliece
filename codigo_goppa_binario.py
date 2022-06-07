"""
Códigos Goppa binarios (CGB).

Este módulo implementa los códigos Goppa binarios.

Autor: Antonio Merino Gallardo.
"""

from sage.rings.finite_rings.finite_field_constructor import GF
from sage.matrix.all import matrix, diagonal_matrix
from sage.modules.free_module_element import vector
from sage.all import copy, xgcd, sqrt

def _calcular_matriz_paridad(g, L):
    """
    Calcula la matriz de paridad de un CGB sobre el cuerpo finito GF(2^m).
    
    Calcula la matriz de paridad de un CGB dado por el polinomio 
    generador g y con conjunto de definición L. Las entradas de la 
    matriz serán elementos del cuerpo finito GF(2^m) en el que están 
    los elementos de L.

    La matriz tendrá dimensión (t x n) con t = g.degree() y n = |L|.
    """
    t = g.degree()
    n = len(L)

    X = matrix([[g.list()[j] for j in range(t-i,t+1)] 
                + [0]*(t-1-i) for i in range(t)])
    Y = matrix([[L[j]**i for j in range(n)] for i in range(t)])
    Z = diagonal_matrix([g(a).inverse_of_unit() for a in L])
    return X*Y*Z

def _calcular_matriz_generadora(H):
    """
    Calcula la matriz generadora de un CGB con matriz de paridad H.
    """
    return H.right_kernel().basis_matrix()

def _descomponer(p):
    """
    Descompone un polinomio p sobre GF(2^m) como p(x) = p0(x)^2 + x*p1(x)^2.
    """
    PR = p.parent()
    raices = [sqrt(c) for c in p.list()]
    return PR(raices[0::2]), PR(raices[1::2])
    
def _extender_matriz(H1):
    """
    Extiende matriz sobre GF(2^m) a matriz sobre GF(2).
    
    Si H1 es una matriz de dimensión (t x n) con elementos sobre el 
    cuerpo finito GF(2^m), la matriz H2 devuelta será una matriz de 
    dimensión (tm x n) con elementos sobre el cuerpo finito GF(2), 
    extendiendo en columna los elementos del cuerpo finito GF(2^m).
    """
    t = H1.nrows()
    n = H1.ncols()
    m = len(vector(H1[0][0]))

    H2 = matrix(GF(2), m*t, 1)

    for i in range(n):
        columna = vector(H1[0][i]).column()
        for j in range(1,t):
            v = vector(H1[j][i]).column()
            columna = columna.stack(v)
        H2 = H2.augment(columna)

    H2 = H2.delete_columns([0])

    return H2

def _invertir(p, g):
    """
    Invierte p en el anillo cociente sobre <g>.
    """
    return xgcd(p, g)[1].mod(g)

def _sqrt(p, g):
    """
    Calcula la raíz cuadrada de p en el anillo cociente sobre <g>.
    """
    g1, g2 = _descomponer(g)
    w = (g1*_invertir(g2, g)).mod(g)
    p1, p2 = _descomponer(p)
    return (p1+w*p2).mod(g)

def _xgcd_grado_acotado(p,q):
    """
    Calcula (d,u,v) tal que d = u*p + v*q acotando los grados de d y u.

    Aplica el algoritmo extendido de Euclides para obtener el máximo
    común divisor y los coeficientes de Bezout para dos polinomios p y
    q acotando el grado del máximo común divisor y de uno de los
    coeficientes de Bezout.

    Devuelve: (d,u,v) tal que: 
        - d = u*p + v*q,
        - d.degree() <= q.degree()//2 y
        - u.degree() <= (q.degree()-1)//2.
    """
    t = q.degree()
    PR = p.parent()
    
    (d1, d2) = (p, q)
    (u1, u2) = (PR(1), PR(0))
    (v1, v2) = (PR(0), PR(1))
    
    while (d1.degree() > t//2) or (u1.degree() > (t-1)//2):
        quotient = d1 // d2
        (d1, d2) = (d2, d1 - quotient * d2)
        (u1, u2) = (u2, u1 - quotient * u2)
        (v1, v2) = (v2, v1 - quotient * v2)
    
    return (d1, u1, v1)


class CodigoGoppaBinario():
    """
    Clase que representa un código Goppa binario.
    """

    def __init__(self, pol_generador, conjunto):
        """
        Constructor de la clase CodigoGoppaBinario.

        Argumentos:
            - pol_generador: polinomio generador del código Goppa Binario.
              Debe estar definido sobre un cuerpo finito de característica 2.
            - conjunto: conjunto de definición del código Goppa Binario.
              Sus elementos no pueden ser raíces de pol_generador.
        """
        self._longitud = len(conjunto)
        self._pol_generador = pol_generador
        self._conjunto = conjunto
        
        F = pol_generador.base_ring().prime_subfield()
        
        if F != GF(2):
            raise ValueError("el polinomio generador debe estar definido sobre un cuerpo finito de característica 2")
        for a in conjunto:
            if pol_generador(a) == 0:
                raise ValueError("los elementos del conjunto no pueden ser raíces del polinomio generador")
        
        self._matriz_paridad = _calcular_matriz_paridad(pol_generador, conjunto)
        self._matriz_paridad_extendida = _extender_matriz(self._matriz_paridad)
        self._matriz_generadora = _calcular_matriz_generadora(self._matriz_paridad_extendida)
        self._dimension = self._matriz_generadora.nrows()
            
    def __repr__(self):
        """
        Devuelve la representación del código Goppa binario.
        """
        return "[{}, {}] Código Goppa Binario".format(self.longitud(), self.dimension())
    
    def longitud(self):
        """
        Devuelve la longitud del código Goppa binario.
        """
        return self._longitud
    
    def dimension(self):
        """
        Devuelve la dimensión del código Goppa binario.
        """
        return self._dimension
    
    def matriz_paridad(self):
        """
        Devuelve la matriz de paridad del código Goppa binario.

        Si t:=_pol_generador.degree(), n:=len(conjunto) y conjunto es 
        un subconjunto de GF(2^m), entonces la matriz de paridad 
        devuelta será una matriz de dimensión (t x n) con sus entradas 
        en GF(2^m).
        """
        return self._matriz_paridad
    
    def matriz_paridad_extendida(self):
        """
        Devuelve la matriz de paridad extendida del código Goppa binario.

        Si t:=_pol_generador.degree(), n:=len(conjunto) y conjunto es 
        un subconjunto de GF(2^m), entonces la matriz de paridad 
        devuelta será una matriz de dimensión (tm x n) con sus entradas 
        en GF(2).
        """
        return self._matriz_paridad_extendida
    
    def matriz_generadora(self):
        """
        Devuelve la matriz generadora del código Goppa binario.
        """
        return self._matriz_generadora
    
    def codificar(self, x):
        """
        Codifica un elemento x en una palabra del código Goppa binario.
        """
        return x*self._matriz_generadora
    
    def pol_sindrome(self, y):
        """
        Devuelve el polinomio síndrome asociado a una palabra y.
        """
        x = self._pol_generador.parent().gen()
        coefs = self._matriz_paridad*y
        
        return sum([coefs[i]*x**(len(coefs)-i-1) for i in range(len(coefs))])
    
    def decodificar(self, y):
        """
        Devuelve la palabra código más cercana a y.

        Aplica el algoritmo de Patterson.
        """
        g = self._pol_generador
        x = g.parent().gen()
        L = self._conjunto
        palabra = copy(y)
        
        # 1. Calculamos polinomio síndrome.
        pol_sind = self.pol_sindrome(y)
        if pol_sind == 0:
            return palabra

        # 2. Calculamos el inverso del polinomio síndrome.
        T = _invertir(pol_sind, g)
        
        # 3.a. Si T(x)=x, el polinomio localizador de errores es x.
        if T == x:
            sigma = x
        else:
            # 3.b. Si T(x)!=x, calculamos la raíz cuadrada de T(x)+x.
            tau = _sqrt(T+x, g)

            # 4. Calculamos alpha(x) y beta(x).
            (alpha, beta, v) = _xgcd_grado_acotado(tau, g)

            # 5. Obtenemos el polinomio localizador de errores sigma(x).
            sigma = alpha**2 + x*beta**2
        
        # 6. Las raíces de sigma(x) indican las posiciones a corregir.
        for i in range(len(L)):
            if sigma(L[i]) == 0:
                palabra[i] +=1
        
        return palabra
        