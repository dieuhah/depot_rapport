#!/usr/bin/env python
# coding: utf-8

# In[77]:


import random


# In[78]:


class Player: # Abstract base class
    
    def __init__(self):
        self.profits = 0
        self.card = None
        
    def set_card( self, card):
        self.card = card
    
    def get_bet( self):
        raise NotImplementedError("Please Implement this method")
    
    def get_action( self, bet ):
        raise NotImplementedError("Please Implement this method")
    
    def get_name( self):
        raise NotImplementedError("Please Implement this method")
    
    def set_results( self, role, card1, card2, bet, action, result ):
        pass
    
    def add_profits( self, profit ):
        self.profits += profit
    
    def get_profits( self ):
        return self.profits
        


# In[79]:


class HeavyPlayer(Player):
    
    def __init__( self ):
        super().__init__()
      
    def get_bet( self ): # I'm sure that I'm 'first' player
        return 5
        
    def get_action( self, bet ): # I'm sure that I'm 'second' player
        return 'go'
    
    def get_name( self ): 
        return 'Heavy'


# In[80]:


class RandomPlayer(Player):
    
    def __init__( self ):
        super().__init__()
      
    def get_bet( self ): # I'm sure that I'm 'first' player 
        return int(5 * random.random() * self.card)
        
    def get_action( self, bet ): # I'm sure that I'm 'second' player 
        if 5 * random.random() * self.card > bet:
            return 'go'
        return 'reject'
      
    def get_name( self ): 
        return 'Random'


# In[81]:


class DeterministicPlayer(Player):
    
    def __init__( self ):
        super().__init__()
      
    def get_bet( self ): # I'm sure that I'm 'first' player
        if self.card > 0.5:
            return 5
        else:
            return 0
        
    def get_action( self, bet ): # I'm sure that I'm 'second' player 
        if self.card > 0.5 or bet == 0:
            return 'go'
        return 'reject'
      
    def get_name( self ): 
        return 'Det'


# In[82]:



def round( player1, player2):
    card1 = random.random()
    card2 = random.random()

    player1.set_card(card1)
    player2.set_card(card2)

    bet = player1.get_bet()
    answer= player2.get_action(bet)
    if bet == 0:
        player2.add_profits( 1 )
        player1.add_profits( -1 )
        gain=-1
        player1.set_results('first', card1, card2, bet, 'go', 'lost')
        player2.set_results('second', card1, card2, bet, 'go', 'won')
    else:
        if answer == 'go':
            if card1 > card2:
                player1.add_profits( 1 + bet )
                player2.add_profits( -1 - bet )
                gain=1 + bet
                player1.set_results('first', card1, card2, bet, 'go', 'won')
                player2.set_results('second', card1, card2, bet, 'go', 'lost')
            else:
                player2.add_profits( 1 + bet )
                player1.add_profits( -1 - bet )
                gain=-1 - bet
                player1.set_results('first', card1, card2, bet, 'go', 'lost' )
                player2.set_results('second', card1, card2, bet, 'go', 'won')
        else:
            player1.add_profits(1)
            player2.add_profits(-1)
            gain=1
            player1.set_results('first', card1, card2, bet, 'reject', 'won')
            player2.set_results('second', card1, card2, bet, 'reject', 'lost')
    #print(player1.get_name(),player1.card,player2.get_name(),player2.card,bet,answer,gain)


# In[99]:


list_espGain = {}
for i in range(1,6):
     list_espGain[i] = ((1+i)*self.card)-((1+i)*(1-self.card))
espMax = max(list_espGain, key=list_espGain.get)
if list_espGain[espMax] > -1:
    valCourbe = m.ceil((m.sqrt(5)*self.card) ** 2)
    dfJ = self.res[df['role']=='first']
    prob = {}
    for i in range(1,6):
        df2J = dfJ[dfJ['bet']==i]
        nb_occu = (df2J['action']=='reject').sum()
        prob[i] = nb_occu/len(df2J)
    maxProbJ = max(prob, key=prob.get)
    return min(maxProbJ, valCourbe)
else:
    return 0


# In[100]:


import numpy as np
import math as m
import pandas as pd

class NotreJoueur(Player):
    def __init__( self ):
        super().__init__()
        self.res = pd.DataFrame(columns=['role', 'card1', 'card2', 'bet', 'action', 'result', 'partC1', 'partC2'])
        self.k = 10
        self.parts = np.split(np.arange(0.0,1.0,0.01), self.k)
        
    def get_res(self):
        return self.res
    
    def get_bet( self ):
        if len(self.res) < 50:
            list_espGain = {}
            for i in range(1,6):
                 list_espGain[i] = ((1+i)*self.card)-((1+i)*(1-self.card))
            espMax = max(list_espGain, key=list_espGain.get)
            if list_espGain[espMax] > -1:
                return m.ceil((m.sqrt(5)*self.card) ** 2)
            else:
                return 0
        else:
            list_espGain = {}
            for i in range(1,6):
                 list_espGain[i] = ((1+i)*self.card)-((1+i)*(1-self.card))
            espMax = max(list_espGain, key=list_espGain.get)
            if list_espGain[espMax] > -1:
                valCourbe = m.ceil((m.sqrt(5)*self.card) ** 2)
                dfJ = self.res[self.res['role']=='first']
                prob = {}
                for i in range(1,6):
                    df2J = dfJ[dfJ['bet']==i]
                    nb_occu = (df2J['action']=='reject').sum()
                    prob[i] = nb_occu/len(df2J)
                maxProbJ = max(prob, key=prob.get)
                return min(maxProbJ, valCourbe)
            else:
                return 0
            
        
    def get_action( self, bet ):
        if bet == 0:
            return 'go'
        else:
            if len(self.res) < 100: # c'est à dire 50 round où l'on joue 2ème
                if self.card > m.sqrt(bet/5) : #on part du principe que le joueur en face utilise la même stratégie que notre joueur A équation inverse de la courbe
                    return 'go'
                else:
                    return 'reject'
            else:
                list_prob2 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                df = self.res[self.res['role']=='second']
                for i in range(self.k):
                    #calcul p(x=Ii) proba soit dans l'intervalle i
                    nb_occuRange = (df['partC1']==i+1).sum()
                    probRange = nb_occuRange/len(df)
                    #calcul de p(M=bet) proba qu'il ait misé une telle mise
                    nb_occuBet = (df['bet']==bet).sum()
                    probBet = nb_occuBet/len(df)
                    #calcul p(x=Ii|M=bet) proba qu'il soit dans l'intervalle i sachant mise m 
                    df2 = df[df['partC1']==i+1]
                    nb_occu = (df2['bet']==bet).sum()
                    prob1 = nb_occu/len(df2)
                    #calcul p(M=bet|x=Ii) avec bayes  proba qu'il ait misé m sachant qu'il est dans intervalle i
                    prob2 = (prob1 * probRange)/probBet
                    list_prob2[i+1] = prob2
                    # trouver dans quelle intervalle se trouve notre propre carte pour la comparer ensuite avec l'autre
                for i in range(self.k):
                    if self.card >= self.parts[i][0] and self.card < self.parts[i][-1]+0.01:
                        partCard = i+1
                somme = 0
                #Calcul p(gain) 
                #somme de tous les intervalles qui sont inférieur ou égal à notre intervalle
                for j in range(1,partCard+1):
                    somme += list_prob2[j]
                probGain = somme 
                #Calcul espérance de gain sur notre carte et celle de la carte inverse et on fait le min des 2 
                esp = ((1+bet)*probGain)-((1+bet)*(1-probGain))
                notreEsp = ((1+bet)*self.card)-((1+bet)*(1-self.card))
                if min(esp, notreEsp) > -1:
                    return 'go'
                else:
                    return 'reject'
        
    def set_results( self, role, card1, card2, bet, action, result ):
        for i in range(self.k):
            if card2 >= self.parts[i][0] and card2 < self.parts[i][-1]+0.01:
                partC2=i+1
            if card1 >= self.parts[i][0] and card1 < self.parts[i][-1]+0.01:
                partC1=i+1
        self.res = self.res.append({'role': role, 'card1': card1, 'card2': card2, 'bet': bet, 'action': action, 'result': result, 'partC1': partC1, 'partC2': partC2}, ignore_index=True)

    def get_name( self ):
        return 'NotreJoueur'


# In[101]:



player2 = DeterministicPlayer()
player1 = NotreJoueur()

for i in range(1000):
    round( player1, player2 )
    #print('-------------------------------------------------------------')
    round( player2, player1 )
    #print('-------------------------------------------------------------')
    
print( 'player1:',player1.get_name(), player1.get_profits())
print( 'player2:',player2.get_name(), player2.get_profits())


# In[102]:


bet = 5
probGain = 0.23
print('Joueur 1:',((1+bet)*probGain)-((1+bet)*(1-probGain)))
print('Joueur 2:',((1+bet)*(1-probGain))-((1+bet)*probGain))


# In[103]:


df = player1.get_res()
df = df[df['role']=='first']
#df2 = df[df['bet']==3]
#print(df2)
prob = {}
for i in range(1,6):
    df2 = df[df['bet']==i]
    nb_occu = (df2['action']=='reject').sum()
    prob[i] = nb_occu/len(df2)
print(prob)


# In[104]:


print(player1.get_res())


# In[117]:


#Faire jouer les ia les unes contre les autres
def tournament(players, nb_rounds=1000):
    results = []
    for x in range(0, len(players)):
        r = []
        for y in range(0, len(players)):
            p1 = players[x]()
            p2 = players[y]()
            for i in range(nb_rounds):
                round( p1, p2 )
                round( p2, p1 )
            r.append(p1.get_profits()/nb_rounds)
        results.append(r)
    return results


# In[118]:


import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

#il vaut mieux commenter le print des résultats de chaque tour pour calculer plus de rounds
players = [HeavyPlayer,
           DeterministicPlayer, 
           RandomPlayer,
           NotreJoueur,
           NotreJoueur2]
labels = list(map(lambda x: x.get_name(x), players))
results = tournament(players, nb_rounds=1000)


# In[119]:


#Résultats du joueur en horizontal face aux joueurs en vertical
ax = sns.heatmap(results, center=0, vmin=-5, vmax=5,linewidths=.5,cmap="seismic_r", xticklabels=labels, yticklabels=labels)


# In[94]:


import numpy as np
import math as m
import pandas as pd

class NotreJoueur2(Player):
    def __init__( self ):
        super().__init__()
        self.res = pd.DataFrame(columns=['role', 'card1', 'card2', 'bet', 'action', 'result', 'partC1', 'partC2'])
        self.k = 10
        self.parts = np.split(np.arange(0.0,1.0,0.01), self.k)
        
    def get_res(self):
        return self.res
    
    def get_bet( self ):
        list_espGain = {}
        for i in range(1,6):
             list_espGain[i] = ((1+i)*self.card)-((1+i)*(1-self.card))
        espMax = max(list_espGain, key=list_espGain.get)
        if list_espGain[espMax] > -1:
            return m.ceil((m.sqrt(5)*self.card) ** 2)
        else:
            return 0
        
    def get_action( self, bet ):
        if bet == 0:
            return 'go'
        else:
            if len(self.res) < 100: # c'est à dire 50 round où l'on joue 2ème
                if self.card > m.sqrt(bet/5) : #on part du principe que le joueur en face utilise la même stratégie que notre joueur A
                    return 'go'
                else:
                    return 'reject'
            else:
                list_prob2 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                df = self.res[self.res['role']=='second']
                for i in range(self.k):
                    #calcul p(x=Ii)
                    nb_occuRange = (df['partC1']==i+1).sum()
                    probRange = nb_occuRange/len(df)
                    #calcul de p(M=bet)
                    nb_occuBet = (df['bet']==bet).sum()
                    probBet = nb_occuBet/len(df)
                    #calcul p(x=Ii|M=bet)
                    df2 = df[df['partC1']==i+1]
                    nb_occu = (df2['bet']==bet).sum()
                    prob1 = nb_occu/len(df2)
                    #calcul p(M=bet|x=Ii) avec bayes
                    prob2 = (prob1 * probRange)/probBet
                    list_prob2[i+1] = prob2
                #maxProb = max(list_prob2, key=list_prob2.get)
                for i in range(self.k):
                    if self.card >= self.parts[i][0] and self.card < self.parts[i][-1]+0.01:
                        partCard = i+1
                somme = 0
                #Calcul p(gain)
                for j in range(1,partCard+1):
                    somme += list_prob2[j]
                probGain = somme
                #Calcul espérance de gain
                esp = ((1+bet)*probGain)-((1+bet)*(1-probGain))
                notreEsp = ((1+bet)*self.card)-((1+bet)*(1-self.card))
                if min(esp, notreEsp) > -1:
                    return 'go'
                else:
                    return 'reject'
        
    def set_results( self, role, card1, card2, bet, action, result ):
        for i in range(self.k):
            if card2 >= self.parts[i][0] and card2 < self.parts[i][-1]+0.01:
                partC2=i+1
            if card1 >= self.parts[i][0] and card1 < self.parts[i][-1]+0.01:
                partC1=i+1
        self.res = self.res.append({'role': role, 'card1': card1, 'card2': card2, 'bet': bet, 'action': action, 'result': result, 'partC1': partC1, 'partC2': partC2}, ignore_index=True)

    def get_name( self ):
        return 'NotreJoueur2'


# In[ ]:




