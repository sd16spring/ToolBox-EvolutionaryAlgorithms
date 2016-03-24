"""
Evolutionary algorithm, attempts to evolve a given message string.

Uses the DEAP (Distributed Evolutionary Algorithms in Python) framework,
http://deap.readthedocs.org

Usage:
    python evolve_text.py [goal_message]

Full instructions are at:
https://sites.google.com/site/sd15spring/home/project-toolbox/evolutionary-algorithms
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
        # See also: http://stackoverflow.com/questions/1436703
        template = '{cls}({val!r})'
        return template.format(cls=self.__class__.__name__,     # "Message"
                               val=self.get_text())

    def get_text(self):
        """Return Message as string (rather than actual list of characters)"""
        return "".join(self)


#-----------------------------------------------------------------------------
# Genetic operators
#-----------------------------------------------------------------------------

#memoization storage
lev_memo = {}
def levenshtein_distance(s1,s2):
    """
    Calculates the levenshtein distance using dynamic programming
    """
    #tuple for memoization key
    key = tuple(sorted((s1,s2)))
    #check memo
    if key in lev_memo.keys():
        return lev_memo[key]
    #creates a row of the string index at the top; these serve as the initial score for the string, as you move away 
    #from the beginning of string 2
    grid = [list(range(len(s2)+1))]
    #create a row for every index in string 1
    for c in s1:
        grid.append([0]*(len(s2)+1))
    #creates a col of the string index on the left; these serve as the initial score for the string as youmove away
    #from the beginning of string 1
    for i in range(len(s1)+1):
        grid[i][0]=i
    #print_grid(grid)
    #Calculate the score for each square in the grid by adding the smallest score before it + cost
    for i in range(1,len(s1)+1):
        for j in range(1,len(s2)+1):
            #if the letters are the same, you don't have to substitute, so cost = 0
            cost = abs(ord(s1[i-1])-ord(s2[j-1])) if s1[i-1]!=s2[j-1] else 0
            #cost+smallest score out of the three previous possible positions
            grid[i][j] = min(grid[i-1][j-1]+cost, grid[i][j-1]+1,grid[i-1][j]+1)
    #get score, the value at the last index of both strings
    score = grid[-1][-1]
    #print_grid(grid)
    #store in memo
    lev_memo[key]=score
    #return
    return score
def print_grid(grid):
    for row in grid:
        for col in row:
            print str(col),
        print 

def two_point_crossover(parent1, parent2):
    """
    Returns the two point crossover of two strings
    """
    length = min(len(parent1), len(parent2))
    points = sorted([random.randrange(length), random.randrange(length)])
    str1 = parent1[:points[0]]+parent2[points[0]:points[1]]+parent1[points[1]:]
    str2 = parent2[:points[0]]+parent1[points[0]:points[1]]+parent2[points[1]:]
    return (Message(starting_string=''.join(str1)), Message(starting_string=''.join(str2)))

def evaluate_text(message, goal_text, verbose=VERBOSE):
    """
    Given a Message and a goal_text string, return the Levenshtein distance
    between the Message and the goal_text as a length 1 tuple.
    If verbose is True, print each Message as it is evaluated.
    """
    distance = levenshtein_distance(message.get_text(), goal_text)
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
        index = random.randrange(len(message))
        message.insert(index, VALID_CHARS[random.randrange(len(VALID_CHARS))])
    if random.random() < prob_del:
        index = random.randrange(len(message))
        del message[index]
    if random.random() < prob_sub:
        index = random.randrange(len(message))
        message[index] = VALID_CHARS[random.randrange(len(VALID_CHARS))]

    return (message, )   # Length 1 tuple, required by DEAP


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
    toolbox.register("mate", two_point_crossover)
    toolbox.register("mutate", mutate_text)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # NOTE: You can also pass function arguments as you define aliases, e.g.
    #   toolbox.register("individual", Message, max_length=200)
    #   toolbox.register("mutate", mutate_text, prob_sub=0.18)

    return toolbox


def evolve_string(text, prob_mate = 0.5, num_gen = 500, prob_mutate = 0.2, population = 300):
    """Use evolutionary algorithm (EA) to evolve 'text' string"""

    # Set random number generator initial seed so that results are repeatable.
    # See: https://docs.python.org/2/library/random.html#random.seed
    #      and http://xkcd.com/221
    random.seed(4)

    # Get configured toolbox and create a population of random Messages
    toolbox = get_toolbox(text)
    pop = toolbox.population(n=population)

    # Collect statistics as the EA runs
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Run simple EA
    # (See: http://deap.gel.ulaval.ca/doc/dev/api/algo.html for details)
    pop, log = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=prob_mate,    # Prob. of crossover (mating)
                                   mutpb=prob_mutate,   # Probability of mutation
                                   ngen=num_gen,    # Num. of generations to run
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
        goal = "SKYNET IS NOW ONLINE"
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
    pop, log = evolve_string(goal, num_gen=len(goal)*60,prob_mutate=.4)
