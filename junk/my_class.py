from my_subclass import first_subclass
from my_listclass import List

class first_class:
	def __init__(self, argument): # executes on instantiation
		self.arg = argument
		self.twice_arg = argument*2
		self.thrice_arg = argument*3
		self.sub_instance = first_subclass()
		self.list = List()
	
	def first_function(self):
		return self.arg*5
	
	def second_function(self, argument):
		return self.arg*argument
	
	def do_stuff(self)
	
