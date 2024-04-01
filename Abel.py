import numpy as np

def Jmean(I, p, border):
     N = 0
     Jnew = np.zeros((2 * p))
     Jnew[0] = Jnew[p * 2 - 1] = (I[0] + I[p * 2 - 1]) / 2
     for i in range(1, p):
          Jnew [i] = Jnew[p * 2 - 1 - i] = (I[i] + I[p * 2 - 1 - i])/2
          if Jnew[i] - Jnew[i - 1] < border/p:
              N = i + 1
              break
     for i in range(N, p):
          Jnew [i] = Jnew[p * 2 - 1 - i] = Jnew [i - 1]
     return Jnew     

def Jmean2ver(I, x, p):
     Jnew = np.zeros((2 * p))
     Xnew = np.zeros((2 * p))
     for i in range(0, p):
          Jnew [i] = Jnew[p * 2 - 1 - i] = (I[i] + I[p * 2 - 1 - i])/2
     centr = (x[0] + x[2 * p - 1]) / 2
     for i in range(0, p):
          Xnew [i] = centr - (x[i] - x[p * 2 - 1 - i])/2
          Xnew[p * 2 - 1 - i] = centr + (x[i] - x[p * 2 - 1 - i])/2
     return Xnew, Jnew

def MatrixL(p, a, k):
    sigmaR = a / p
    L = np.zeros((p, k))
    Lreshape = np.zeros ((p, k))
    for i in range(p):
         for j in range(k):
                 if ((i + 1) ** 2 - j ** 2) < 0 or (i ** 2 - j ** 2) < 0:
                       continue
                 L[i, j] = (2 * sigmaR * (np.sqrt((i + 1) ** 2 - j ** 2) - np.sqrt(i ** 2 - j ** 2))).real
                 Lreshape[p - 1 - i, k - 1 - j] = L[i, j]
    return Lreshape

def AbeL(p, k, L, I):
      A = np.zeros((p))
      for i in range(p):
         for j in range(k):
              A[i] += I[i] * L[i, j]
      return A

def gauss(x, A, sigma): 
    return A * np.exp(-( 2 * np.sqrt(np.log(2)) * x / sigma) ** 2)

def gaussSum(x, A, sigma, k):
     Sum = 0
     for i in range(0, k):
          Sum += gauss(x, A[i], sigma[i])
     return Sum