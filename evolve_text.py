"""
Evolutionary algorithm, attempts to evolve a given message string.
Uses the DEAP (Distributed Evolutionary Algorithms in Python) framework,
http://deap.readthedocs.org
Usage:
    python evolve_text.py [goal_message]

@author: March Saper

Written for Software Design 2016 - Olin College of Engineering

"""

import random
import string

import numpy    # Used for statistics
from deap import algorithms
from deap import base
from deap import tools


#-----------------------------------------------------------------------------
# Global variables
#-----------------------------------------------------------------------------

# Allowable characters include all uppercase letters and space
# You can change these, just be consistent (e.g. in mutate operator)
VALID_CHARS = string.ascii_uppercase + " "

# Dictionary to memoize for levenstein distance, thus significantly decreasing run time
LEV_DICT = {}

# Control whether all Messages are printed as they are evaluated
VERBOSE = True


#-----------------------------------------------------------------------------
# Message object to use in evolutionary algorithm
#-----------------------------------------------------------------------------

class FitnessMinimizeSingle(base.Fitness):
    """
    Class representing the fitness of a given individual, with a single
    objective that we want to minimize (weight = -1)
    """
    weights = (-1.0, )


class Message(list):
    """
    Representation of an individual Message within the population to be evolved

    We represent the Message as a list of characters (mutable) so it can
    be more easily manipulated by the genetic operators.
    """
    def __init__(self, starting_string=None, min_length=4, max_length=30):
        """
        Create a new Message individual.

        If starting_string is given, initialize the Message with the
        provided string message. Otherwise, initialize to a random string
        message with length between min_length and max_length.
        """
        # Want to minimize a single objective: distance from the goal message
        self.fitness = FitnessMinimizeSingle()

        # Populate Message using starting_string, if given
        if starting_string:
            self.extend(list(starting_string))

        # Otherwise, select an initial length between min and max
        # and populate Message with that many random characters
        else:
            initial_length = random.randint(min_length, max_length)
            for i in range(initial_length):
                self.append(random.choice(VALID_CHARS))

    def __repr__(self):
        """Return a string representation of the Message"""
        # Note: __repr__ (if it exists) is called by __str__. It should provide
        #       the most unambiguous representation of the object possible, and
        #       ideally eval(repr(obj)) == obj
        template = '{cls}({val!r})'
        return template.format(cls=self.__class__.__name__,     # "Message"
                               val=self.get_text())

    def get_text(self):
        """Return Message as string (rather than actual list of characters)"""
        return "".join(self)


#-----------------------------------------------------------------------------
# Genetic operators
#-----------------------------------------------------------------------------

def lev(a, b):
    """ Returns the Levenshtein distance between strings a and b

        >>> lev("apple", "opple")
        1
        >>> lev("kitten", "smitten")
        2
        >>> lev("software", "software")
        0
    """

    # Base cases: one of the strings is empty
    if (a, b) in LEV_DICT:
        return LEV_DICT[(a,b)]

    if a == "":
        return len(b)
    if b == "":
        return len(a)

    # Strategy 1: Change the first character to match
    if a[0] == b[0]:
        # First character already matches, no extra distance
        option1 = lev(a[1:], b[1:])
    else:
        # First character is different, distance of 1 to change it
        option1 = 1 + lev(a[1:], b[1:])

    # Strategy 2: Insert b[0] as the first character of a
    option2 = 1 + lev(a, b[1:])

    # Strategy 3: Delete the first character of a
    option3 = 1 + lev(a[1:], b)

    distance = min(option1, option2, option3)

    LEV_DICT[(a,b)] = distance

    return distance


def evaluate_text(message, goal_text, verbose=VERBOSE):
    """
    Given a Message and a goal_text string, return the Levenshtein distance
    between the Message and the goal_text as a length 1 tuple.
    If verbose is True, print each Message as it is evaluated.
    """
    distance = lev(message.get_text(), goal_text)
    if verbose:
        print "{msg:60}\t[Distance: {dst}]".format(msg=message, dst=distance)
    return (distance, )     # Length 1 tuple, required by DEAP


def mutate_text(message, prob_ins=0.05, prob_del=0.05, prob_sub=0.05):
    """
    Given a Message and independent probabilities for each mutation type,
    return a length 1 tuple containing the mutated Message.

    Possible mutations are:
        Insertion:      Insert a random (legal) character somewhere into
                        the Message
        Deletion:       Delete one of the characters from the Message
        Substitution:   Replace one character of the Message with a random
                        (legal) character
    """

    if random.random() < prob_ins:
        insert_i = random.randint(0, len(message)-1)
        insert_char = random.choice(VALID_CHARS)
        message.insert(insert_i, insert_char)

    if random.random() < prob_del:
    	delete_i = random.randint(0, len(message)-1)
    	del message[delete_i]

    if random.random() < prob_sub:
    	sub_i = random.randint(0, len(message)-1)
    	sub_char = random.choice(VALID_CHARS)
    	message[sub_i] = sub_char

    return (Message(message), ) # Length 1 tuple, required by DEAP
    
def mate_text(parent1, parent2):
	"""
	Given two parent input strings, returns two child strings which are a 
    crossover of their parents. 
	First, finds determines the indices on which to mate the parents strings.
	Second, creates the child strings.
	"""

	mutate_point1 = random.randint(0,len(parent1)-1)
	mutate_point2 = random.randint(mutate_point1 + 1, len(parent1))

	child1 = parent1[:mutate_point1] + parent2[mutate_point1:mutate_point2] + parent1[mutate_point2:]
	child2 = parent2[:mutate_point1] + parent1[mutate_point1:mutate_point2] + parent2[mutate_point2:]

	return (child1, child2)



#-----------------------------------------------------------------------------
# DEAP Toolbox and Algorithm setup
#-----------------------------------------------------------------------------

def get_toolbox(text):
    """Return DEAP Toolbox configured to evolve given 'text' string"""

    # The DEAP Toolbox allows you to register aliases for functions,
    # which can then be called as "toolbox.function"
    toolbox = base.Toolbox()

    # Creating population to be evolved
    toolbox.register("individual", Message)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluate_text, goal_text=text)
    #toolbox.register("mate", mate_text)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_text)
    toolbox.register("select", tools.selTournament, tournsize=3)


    return toolbox


def evolve_string(text):
    """Use evolutionary algorithm (EA) to evolve 'text' string"""

    # Set random number generator initial seed so that results are repeatable.
    random.seed(4)

    # Get configured toolbox and create a population of random Messages
    toolbox = get_toolbox(text)
    pop = toolbox.population(n=100)

    # Collect statistics as the EA runs
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Run simple EA

    pop, log = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=0.5,    # Prob. of crossover (mating)
                                   mutpb=0.2,   # Probability of mutation
                                   ngen=100,    # Num. of generations to run
                                   stats=stats)

    return pop, log


#-----------------------------------------------------------------------------
# Run if called from the command line
#-----------------------------------------------------------------------------

if __name__ == "__main__":

    # Get goal message from command line (optional)
    import sys
    if len(sys.argv) == 1:
        # Default goal of the evolutionary algorithm if not specified.
        # Pretty much the opposite of http://xkcd.com/534
        goal = "HEFFALUMP"
    else:
        goal = " ".join(sys.argv[1:])

    # Verify that specified goal contains only known valid characters
    # (otherwise we'll never be able to evolve that string)
    for char in goal:
        if char not in VALID_CHARS:
            msg = "Given text {goal!r} contains illegal character {char!r}.\n"
            msg += "Valid set: {val!r}\n"
            raise ValueError(msg.format(goal=goal, char=char, val=VALID_CHARS))

    # Run evolutionary algorithm
    pop, log = evolve_string(goal)
