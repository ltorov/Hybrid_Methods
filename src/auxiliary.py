import numpy as np
def partitioning(datos,size_S,size_T):
    """ 
        Método que hace el partitioning de los datos en el set de S, training, testing y validation
        Recibe:
            1. datos: Datos al cual se le va a hacer el partitioning
            2. Size_S: Tamaño del set S
            3. Size_T: Tamaño del set T (training)
        Entrega:
            1. S: Set S
            2. V: set de validación V
            3. T: set de training T
            4. t: set de testing t
    """
    indices_datos = np.arange(len(datos))

    indices_S = np.random.choice(indices_datos, size=int(round(len(datos)*size_S)),replace=False)
    indices_V = np.setdiff1d(indices_datos, indices_S)
    indices_T = np.random.choice(indices_S, size=int(round(len(datos)*size_T)),replace=False)
    indices_t = np.setdiff1d(indices_S, indices_T)

    S= datos[indices_S]
    
    V= datos[np.random.choice(indices_V, size=len(indices_V),replace=False)]
    T= datos[indices_T]
    t= datos[np.random.choice(indices_t, size=len(indices_t),replace=False)]

    return S,V,T,t