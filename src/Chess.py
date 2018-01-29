import chess
import Computer
import random

if __name__ == '__main__':

    board = chess.Board()
    color = input('Please choose computer color:\nw = White\nb = Black\n--> ')
    
    comp = Computer.ComputerPlayer.fromFile('../model/model_full_weights.npz', color)
    comp.setDiff(4, 180, random.randint(1,100), 0, 0)

    print(board)
    
    while board.is_game_over() == False:
        # get white's move
        if comp.color == 1:
            print('Thinking...')
            move = comp.getMove(board.copy())
            print("Computer's best move: " + str(move))
            print()
            board.push(move[0])
        else:
            while True:
                try:
                    move = input('Please enter move in full alegbraic notation\n--> ')
                    board.push_uci(move)
                    break
                except ValueError:
                    print('Invalid move. Try again')
        print(board)
        
        #get black's move
        if comp.color == -1:
            print('Thinking...')
            move = comp.getMove(board.copy())
            board.push(move[0])
            print("Computer's best move: " + str(move))
            print()
        else:
            while True:
                try:
                    move = input('Please enter move in full alegbraic notation\n--> ')
                    board.push_uci(move)
                    break
                except ValueError:
                    print('Invalid move. Try again')
            
        print(board)
