import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.01
        self.alpha = 1

        #creating Q dictionary
        self.Q = {}
        self.Q.clear()

        # initializing states
       #self.state = ["Red", "Red_right", "rGreen_f", "rGreen_r", "Green_l", "Green_f", "Green_r"]
        self.state = None
        self.action = [None, "forward", "left", "right"]

        # initializing previous and next states/actions/rewards
        self.pre_state = None
        self.next_state = None
        self.pre_act = None
        self.next_act = None
        self.pre_reward = 0
        self.next_reward = 0
        self.trainingPassed = 0
        self.trainingFailed = 0
        self.trialsPassed = 0
        self.trialsFailed = 0
        self.totalScore = 0 
        self.totalBadMoves = 0 
        self.totalOKMoves = 0
        self.totalGoodMoves = 0
        self.logBadStates = []

        # initializing trial counts to zero
        self.training_count = 0
        self.trial_count = 0
        
        self.finalTenTrials = {}

        self.typeOfMoves = {}
        self.typeOfMoves['Bad'] = 0
        self.typeOfMoves['Ok'] = 0
        self.typeOfMoves['Good'] = 0
        self.trainedTrials = {}
        self.trainedTrialsScores = {}
        self.trialMovesBad = {}
        self.trialMovesOk = {}
        self.trialMovesGood = {}


        # initializing Q function to all zeros
        # we have total of 512 states in addition to
        self.Q[(None,None)]=0
        self.light_color = ['red', 'green']
        for l in self.light_color:
            for a1 in self.action:
                for a2 in self.action:
                    for a3 in self.action:
                        for a4 in self.action:
                            self.Q[((l,a1,a2,a3),a4)] = 10
       

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print " ---------------------------- Starting new trial -----------------------------"
        self.trial_count += 1
        print " ---------------------------- Trial " +str(self.trial_count) + " -----------------------------"
        self.totalScore = 0
        self.totalBadMoves = 0
        self.totalOKMoves = 0
        self.totalGoodMoves = 0

        

    def update(self, t):
        # Gather inputs
        print " --------------------- update " + str(t) + "----------------------"
        
        myTrial = "Trial"+str(self.trial_count)
        myTrialScore = "Trial"+str(self.trial_count) + "_score"

        # counts number of trials to train the agent
        self.training_count += 1
        print "\nthis is the training count: " + str(self.training_count)

        # saving the next state to previous state before change
        sn = self.next_state
        self.pre_state = sn
        sp = self.pre_state

        # saving the next action to previous action before change
        an = self.next_act
        self.pre_act = an
        ap = self.pre_act

        # saving the next reward to previous reward before change
        rn = self.next_reward
        self.pre_reward = rn
        rp = self.pre_reward

        # Gathering actual inputs from environment
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        wp = self.next_waypoint        
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # Generate next states according to inputs
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], wp)
        
        
        self.next_state = self.state
        sn = self.next_state
        print str(sn) + " next state"
        """
        if inputs['light'] == 'red':
            if inputs['left'] == 'forward':
                self.next_state = self.state[0]
            else:
                self.next_state = self.state[1]  
        if inputs['light'] == 'green':
            if inputs['oncoming'] == 'forward':
                if wp == "forward":
                    self.next_state = self.state[2]
                if wp == "right":
                    self.next_state = self.state[3]
            else:
                if wp == "left":
                    self.next_state = self.state[4]
                if wp == "forward":
                    self.next_state = self.state[5]
                if wp == "right":
                    self.next_state = self.state[6]
        
        sn = self.next_state
        """

        # TODO: Select action according to your policy
        """  This block of code is from our initial random testing
        #possibleActions = [None, 'forward', 'left', 'right']
        # sample input
        # inputs = {'light': 'red', 'oncoming': None, 'right': None, 'left': None}
        #action = random.choice(possibleActions) # changed from none
        
        """
        
        """
        if (sn,an) in self.Q:
            # if state exists updated it
            # finding the maxQ value
            print "am i in wrong side"
            allQactions = []
            for key in self.Q.keys():
                if key[0] == sn:
                    allQactions.append(self.Q[key])
        else:
            # if state doesn't exist initial lize to 0
            self.Q[(sn,an)] = 0 """
        
        trainingRounds = 2200   #2200


        act = self.action   # saving the list of actions into variable act since it's shorter

        if (self.training_count < trainingRounds):  # chnaged from 500
            # training the agent first with random moves or else we get stuck
            self.next_act = random.choice(self.action)
        else:
            # turning off the training so we can follow greedy actions
            #self.alpha = 1
            #self.gamma = 0
            
            # creating a list of the Q values for the given state for every possible action
            """
            availableQactions = []
            for key in self.Q.keys():
                if key[0] == sn:
                    availableQactions.append(self.Q[key])
            
            print availableQactions
            max_value = max(availableQactions)
            max_index = availableQactions.index(max_value)
            print max_index
            self.next_act = self.action[max_index]
            """
            # creating a list of the Q values for the given state for every possible action
            my_list = (self.Q[(sn, act[0])], self.Q[(sn, act[1])], self.Q[(sn, act[2])], self.Q[(sn, act[3])])
            max_value = max(my_list)
            max_index = my_list.index(max_value)

            # selecting the action with the highest Q value
            self.next_act = self.action[max_index]

        # Execute action and get reward
        an = self.next_act  # saving the nextaction into variable "an" since it's shorter
        self.next_reward = self.env.act(self, an)
        rn = self.next_reward # again saving the next reward into variable "rn" since it's shorter

        # TODO: Learn policy based on state, action, reward
    
        

        print "previous state q value before: " + str(self.Q[(sp,ap)])
        
        print str((sp,ap)) + " previous state"
        # Here is the Q updating with tuning parameters alpha and gamma
        if (sn,an) in self.Q:
            # if state exists updated it
            # finding the maxQ value
            """
            print "am i in wrong side"
            allQactions = []
            for key in self.Q.keys():
                if key[0] == sn:
                    allQactions.append(self.Q[key])
            findMaxQ = max(allQactions)
                    """
            findMaxQ = max(self.Q[(sn, act[0])], self.Q[(sn, act[1])], self.Q[(sn, act[2])], self.Q[(sn, act[3])])

            
            self.Q[(sp,ap)] += self.Q[(sp,ap)] + self.alpha*(rp + self.gamma * findMaxQ - self.Q[(sp,ap)])
        else:
            # if state doesn't exist initial lize to 0
         #   print "hello" + str((sn,an))
        #    print str((sn,act[0]))

            self.Q[(sn,act[0])] = 0
            self.Q[(sn,act[1])] = 0
            self.Q[(sn,act[2])] = 0
            self.Q[(sn,act[3])] = 0
        
        #print self.Q       # debug purposes
        print "\n"
        
        if (self.training_count < trainingRounds):
            print "training ...."
            if rn >= 10:
                self.trainingPassed +=1
            
            if rn == -1:
                self.totalBadMoves += 1
            if rn == -0.5:
                self.totalOKMoves += 1
            if rn == 2:
                self.totalGoodMoves += 1
                
            if deadline == 0:
                self.trainingFailed += 1
        
        if (self.training_count >= trainingRounds):
            print "training complete!"
            if rn >= 10:
                self.trialsPassed +=1
                self.trainedTrials[myTrial] = 'Pass'
            
            if rn == -1:
                self.totalBadMoves += 1
                self.logBadStates.append((sn,an))
            if rn == -0.5:
                self.totalOKMoves += 1
            if rn == 2:
                self.totalGoodMoves += 1
        
            if deadline == 0:
                self.trialsFailed += 1
                self.trainedTrials[myTrial] = 'Fail'

        print "\nTraining passed:  " + str(self.trainingPassed)
        print "Training failed:  " + str(self.trainingFailed)
        print "Trials passed:  " + str(self.trialsPassed)
        print "Trials failed:  " + str(self.trialsFailed)
        
        self.totalScore += rn
        print "\ntotal game score: " + str(self.totalScore)
        if (self.training_count >= trainingRounds):
            self.trainedTrialsScores[myTrialScore] = self.totalScore
            self.trialMovesBad[myTrial] = self.totalBadMoves
            self.trialMovesOk[myTrial] = self.totalBadMoves
            self.trialMovesGood[myTrial] = self.totalGoodMoves
            
        
        #print "previous state q value after: " + str(self.Q[(sp,ap)])
        #print "previous state:  " + str(sp)
        #print "previous action:  " + str(ap)
        
        #print "next state q value after: " + str(self.Q[(sn,an)])
        #print "next state:  " + str(sn)
        #print "next action:  " + str(an)
    
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, state = {}".format(deadline, inputs, an, rn, sn)  # [debug] added state
        #print "\nthis is next" + str(next_inputs)
        print self.trainedTrials
        print self.trainedTrialsScores
        print self.trialMovesBad
        print self.logBadStates
        print self.trialMovesOk
        print self.trialMovesGood

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    #sim = Simulator(e, update_delay=0.001, display=True)  # create simulator (uses pygame when display=True, if available)
    # removed display=True to prevent error
    sim = Simulator(e, update_delay=0.001)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
