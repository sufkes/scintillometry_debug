import os, sys
from my_class import first_class

argument_1 = 2
argument_2 = 7

instance = first_class(argument_1)

print instance.arg, instance.twice_arg, instance.thrice_arg # Access attributes of the instance of the class.

print instance.first_function()

print instance.second_function(argument_2)

print instance.sub_instance.sub_val_1


