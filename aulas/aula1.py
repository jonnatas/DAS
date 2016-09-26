import pytest

class Calculadora(object):
	"""docstring for Calculadora"""
	def __init__(self, arg):
		super(Calculadora, self).__init__()
		self.arg = arg
	
	def soma(self,x):
		#return list(map(lambda i: i+x, self.arg))
		return [i+x for i in self.arg]

	def mux(self,x):
		return [i*x for i in self.arg]

	def div(self,x):
		return [float(i/x) for i in self.arg]

	def minus(self,x):
		return [i-x for i in self.arg]
		
def test_soma():
	calc = Calculadora([0,1,2])
	assert([2,3,4]== calc.soma(2.0))

def test_mux():
	calc = Calculadora([0,1,2])
	assert([0,5,10] == calc.mux(5))

def test_div():
	calc = Calculadora([0,1,2])
	assert([0.0, 0.2, 0.4] == calc.div(5.0))
