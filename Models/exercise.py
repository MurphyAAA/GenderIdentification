player1 = [10,6,8,9,7,12,7]
player2 = [7,6,9,5,2,8,11]
win1 = 0;
win2 = 0;
print ("******Card Busters****** \n")
for i in range(len(player1)):
  if (player1[i] > player2[i]):
    win1+=1
    print("Round number %d: Player %d wins the round, with %d beating %d \n" % (i+1, 1, player1[i],player2[i]))
  elif (player1[i] == player2[i]):
    print("Round number %d: This round has ended in a draw \n" %(i+1))
  else:
    win2+=1
    print("Round number %d:Player %d wins the round, with %d beating %d \n" % (i+1, 2, player2[i],player1[i]))

if (win1>win2):
  print("Player %d wins the game, by %d wins to %d \n" % ( 1, win1 ,win2))
elif (win1==win2):
  print("The game has ended in a tie")
else:
  print("Player %d wins the game, by %d wins to %d \n" % ( 2, win2 ,win1))
